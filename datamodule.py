"""
__author__: Nabin
This is the pytorch lightning datamodule for Yeti.
"""

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
import logging


import biotite.structure.io.pdbx as pdbx_parser

import pandas as pd
import numpy as np
# from Bio.PDB import MMCIFParser
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig

import biotite

logger = logging.getLogger(__name__)


class ProteinStructureDataset(Dataset):
    def __init__(
        self,
        metadata_tsv: str,
        cif_dir: str,
        seq_dir: str,
        plddt_threshold: float = 30.0,
        max_seq_length: Optional[int] = None,
        min_seq_length: Optional[int] = None,
        device: str = 'cpu',
        max_total_samples: Optional[int] = None,
        backbone_atom_mode: str = "CA",
        max_coil_pct: Optional[float] = None,
        min_mean_plddt: Optional[float] = None,
        min_pct_residues_high_plddt: Optional[float] = None,
        save_filtered_metadata_path: Optional[str] = None,
    ):
        self.cif_dir = cif_dir
        self.seq_dir = seq_dir
        self.plddt_threshold = plddt_threshold
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.device = device
        self.max_total_samples = max_total_samples
        self.max_coil_pct = max_coil_pct
        self.min_mean_plddt = min_mean_plddt
        self.min_pct_residues_high_plddt = min_pct_residues_high_plddt
        self.save_filtered_metadata_path = save_filtered_metadata_path
        backbone_atom_mode = (backbone_atom_mode or "CA").upper()
        if backbone_atom_mode not in {"CA", "FULL"}:
            raise ValueError(
                f"Unsupported backbone_atom_mode={backbone_atom_mode}. "
                "Allowed values: 'CA' or 'FULL'."
            )
        self.backbone_atom_mode = backbone_atom_mode

        # Load metadata (CSV or TSV). Infer separator from file extension.
        _sep = ',' if metadata_tsv.rstrip().lower().endswith('.csv') else '\t'
        self.df = pd.read_csv(metadata_tsv, sep=_sep)
        self._filter_data()

        # If max_samples is not None, we only use the first max_samples proteins in the dataset. This is used for testing with small data.
        if max_total_samples is not None:
            self.df = self.df.head(max_total_samples)
            logger.info(f"Loaded the first {max_total_samples} proteins. Check max_total_samples parameter in the config file.")

        if self.save_filtered_metadata_path:
            os.makedirs(os.path.dirname(self.save_filtered_metadata_path) or ".", exist_ok=True)
            _out_sep = "," if self.save_filtered_metadata_path.rstrip().lower().endswith(".csv") else "\t"
            self.df.to_csv(self.save_filtered_metadata_path, sep=_out_sep, index=False)
            logger.info(f"Saved filtered metadata ({len(self.df)} rows) to {self.save_filtered_metadata_path}")

        logger.info(f"Loaded {len(self.df)} protein structures. All Proteins loaded.")

    
    def _filter_data(self):
        initial_count = len(self.df)
        # 1. Remove chains with coil percentage > max_coil_pct (default: keep coil_pct <= 70%)
        if self.max_coil_pct is not None and 'coil_pct' in self.df.columns:
            self.df = self.df[(self.df['coil_pct'].isna()) | (self.df['coil_pct'] <= self.max_coil_pct)]
        # 2. Remove chains with mean residue-wise pLDDT < min_mean_plddt (default: keep mean_plddt >= 80 or 70)
        if self.min_mean_plddt is not None and 'mean_plddt' in self.df.columns:
            self.df = self.df[(self.df['mean_plddt'].isna()) | (self.df['mean_plddt'] >= self.min_mean_plddt)]
        elif self.plddt_threshold is not None and 'mean_plddt' in self.df.columns:
            self.df = self.df[(self.df['mean_plddt'].isna()) | (self.df['mean_plddt'] >= self.plddt_threshold)]
        # 3. Keep only chains where at least min_pct_residues_high_plddt% of residues have high pLDDT (e.g. 80% have pLDDT > 70)
        if self.min_pct_residues_high_plddt is not None and 'pct_residues_high_plddt' in self.df.columns:
            self.df = self.df[(self.df['pct_residues_high_plddt'].isna()) | (self.df['pct_residues_high_plddt'] >= self.min_pct_residues_high_plddt)]
        if self.min_seq_length is not None and self.max_seq_length is not None and 'num_residues' in self.df.columns:
            self.df = self.df[(self.df['num_residues'] >= self.min_seq_length) & (self.df['num_residues'] <= self.max_seq_length)]
        
        logger.info(f"Filtered from {initial_count} to {len(self.df)} structures")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        for key in ('protein_id', 'protein'):
            if key in row.index and pd.notna(row.get(key)):
                name = str(row[key]).split('.')[0]
                break
        else:
            name = str(row.name)

        backbone_coords = self._extract_backbone_coords(name)
        if backbone_coords is None:
            raise RuntimeError(f"Failed to extract backbone coordinates for {name}")

        # load the sequence from the fasta file.
        # not used currently, but the idea is to have a sequence head so the model is aware of the sequence information too... 
        seq = self._load_sequence(name)

        coords_ca_raw = backbone_coords["CA"].astype(np.float32)
        num_bad_ca = np.count_nonzero(~np.isfinite(coords_ca_raw))
        # if there are any non-finite entries in the CA coordinates, replace them with 0.0. Rarely happens.
        if num_bad_ca > 0:
            logger.info(
                f"[CA] Replacing {num_bad_ca} non-finite entries for {name} "
                f"with 0.0 in coords_ca"
            )
        coords_ca = np.nan_to_num(coords_ca_raw, nan=0.0, posinf=0.0, neginf=0.0)
        stacked = backbone_coords["stacked"]
        if self.backbone_atom_mode == "CA":
            # Only require CA to be finite when training CA-only models.
            valid_mask = np.isfinite(coords_ca_raw).all(axis=-1)
        else:
            # FULL mode: require all backbone atoms (N, CA, C, O).
            valid_mask = np.isfinite(stacked).all(axis=(1, 2))

        protein = {
            "name": name,
            "seq": seq,
            "coords": list(
                zip(
                    backbone_coords["N"].tolist(),
                    backbone_coords["CA"].tolist(),
                    backbone_coords["C"].tolist(),
                    backbone_coords["O"].tolist(),
                )
            ),
            "coords_ca": coords_ca,
            "valid_mask": valid_mask.astype(bool),
            "chain_ids": row.get('chain_ids', None),
            "mean_pLLDT_score": row.get('mean_plddt', None),
            "num_residues": len(seq),
        }

        if self.backbone_atom_mode == "FULL":
            num_bad_backbone = np.count_nonzero(~np.isfinite(stacked))
            if num_bad_backbone > 0:
                logger.info(
                    f"[FULL] Replacing {num_bad_backbone} non-finite entries for {name} "
                    f"with 0.0 in coords_backbone"
                )
            coords_backbone = np.nan_to_num(
                stacked.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            protein["coords_backbone"] = coords_backbone

        return protein


    def _extract_backbone_coords(self, name: str) -> Dict[str, np.ndarray]:
        """
        Extract the backbone coordinates from the CIF file. 
        """
        cif_path = f"{self.cif_dir}/{name}.cif"
        if not os.path.isfile(cif_path):
            logger.critical(f"CIF file not found for {name}: {cif_path}")
            return None


        # Read mmCIF file with Biotite and extract backbone atom coordinates
        cif_file = pdbx_parser.CIFFile.read(cif_path)
        structure = pdbx_parser.get_structure(cif_file, model=1)

       
        atom_names = structure.atom_name
        res_ids = structure.res_id
        coords = structure.coord

        # Keep only backbone atoms
        backbone_mask = np.isin(atom_names, ["N", "CA", "C", "O"])
        bb_atom_names = atom_names[backbone_mask]
        bb_res_ids = res_ids[backbone_mask]
        bb_coords = coords[backbone_mask]

        N, CA, C, O = list(), list(), list(), list()

        # Iterate over residues and collect N, CA, C, O coordinates.
        for res_id in np.unique(bb_res_ids):
            mask = bb_res_ids == res_id
            names_res = bb_atom_names[mask]
            coords_res = bb_coords[mask]

            def get_atom_coord(target: str):
                idx = np.where(names_res == target)[0]
                if idx.size > 0:
                    return coords_res[idx[0]].astype(float).tolist()
                else:
                    logger.error(f"No {target} atom found for {name} in {cif_path}. Using nan for coordinates.")
                    return [float("nan")] * 3

            N.append(get_atom_coord("N"))
            CA.append(get_atom_coord("CA"))
            C.append(get_atom_coord("C"))
            O.append(get_atom_coord("O"))

        N = np.asarray(N, dtype=np.float32)
        CA = np.asarray(CA, dtype=np.float32)
        C = np.asarray(C, dtype=np.float32)
        O = np.asarray(O, dtype=np.float32)

        stacked = np.stack([N, CA, C, O], axis=1)  # (L, 4, 3)
        return {
            "N": N,
            "CA": CA,
            "C": C,
            "O": O,
            "stacked": stacked,
        }
  
    def _load_sequence(self, name: str) -> str:
        """
        Parse the fasta file and return the sequence.
        """
        seq_path = f"{self.seq_dir}/{name}.fasta"
        with open(seq_path, 'r') as f:
            lines = f.readlines()
            sequence = ''.join(line.strip() for line in lines[1:])
            return sequence


class YetiDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers

        self.metadata_tsv = cfg.data.meta_data
        self.cif_dir = cfg.data.cif_dir
        self.seq_dir = cfg.data.seq_dir

        self.plddt_threshold = getattr(cfg.data, 'plddt_threshold', None)
        self.max_seq_length = getattr(cfg.data, 'max_seq_length', None)
        self.min_seq_length = getattr(cfg.data, 'min_seq_length', None)
        self.backbone_atom_mode = getattr(cfg.data, 'backbone_atom_mode', 'CA')

        self.train_split = cfg.data.train_split
        self.val_split = cfg.data.val_split
        self.test_split = cfg.data.test_split

        self.max_train_samples = getattr(cfg.data, 'max_train_samples', None)
        self.max_val_samples = getattr(cfg.data, 'max_val_samples', None)
        self.max_test_samples = getattr(cfg.data, 'max_test_samples', None)

        self.max_coil_pct = getattr(cfg.data, 'max_coil_pct', None)
        self.min_mean_plddt = getattr(cfg.data, 'min_mean_plddt', None)
        self.min_pct_residues_high_plddt = getattr(cfg.data, 'min_pct_residues_high_plddt', None)
        self.save_filtered_metadata_path = getattr(cfg.data, 'save_filtered_metadata_path', None)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_dataset = ProteinStructureDataset(
                metadata_tsv=self.metadata_tsv,
                cif_dir=self.cif_dir,
                seq_dir=self.seq_dir,
                plddt_threshold=self.plddt_threshold,
                device=self.cfg.data.device,
                max_total_samples=getattr(self.cfg.data, 'max_total_samples', None),
                max_seq_length=self.max_seq_length,
                min_seq_length=self.min_seq_length,
                backbone_atom_mode=self.backbone_atom_mode,
                max_coil_pct=self.max_coil_pct,
                min_mean_plddt=self.min_mean_plddt,
                min_pct_residues_high_plddt=self.min_pct_residues_high_plddt,
                save_filtered_metadata_path=self.save_filtered_metadata_path,
            )

            total_size = len(full_dataset)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size

            generator = torch.Generator().manual_seed(self.cfg.train.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=generator
            )

            # if self.max_train_samples is not None:
            # train_indices = list(range(min(len(self.train_dataset), self.max_train_samples)))
            train_indices = list(range(len(self.train_dataset)))
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices)

            if self.cfg.data.save_train_ids:
                # comment the below while running for entire dataset or not. This is just for testing with small data and to see which proteins are used.
                with open(self.cfg.data.save_train_ids_path, 'w') as f:
                    for i in range(min(200, len(self.train_dataset))):
                        item = self.train_dataset[i]
                        f.write(f"{item['name']}\n")

            # if self.max_val_samples is not None:
            # val_indices = list(range(min(len(self.val_dataset), self.max_val_samples)))
            val_indices = list(range(len(self.val_dataset)))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices)

            if self.cfg.data.save_val_ids:
                with open(self.cfg.data.save_val_ids_path, 'w') as f:
                    for i in range(min(200, len(self.val_dataset))):
                        item = self.val_dataset[i]
                        f.write(f"{item['name']}\n")
                    val_indices = list(range(len(self.val_dataset)))


            test_indices = list(range(len(self.test_dataset)))        
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, test_indices)

            if self.cfg.data.save_test_ids:
                with open(self.cfg.data.save_test_ids_path, 'w') as f:
                    for i in range(min(200, len(self.test_dataset))):
                        item = self.test_dataset[i]
                        f.write(f"{item['name']}\n")

        print(f"[Info] train_dataset: {len(self.train_dataset)}")
        print(f"[Info] val_dataset: {len(self.val_dataset)}")
        print(f"[Info] test_dataset: {len(self.test_dataset)}")

        if stage == "test" or stage is None:
            if not hasattr(self, 'test_dataset'):
                full_dataset = ProteinStructureDataset(
                    metadata_tsv=self.metadata_tsv,
                    cif_dir=self.cif_dir,
                    seq_dir=self.seq_dir,
                    plddt_threshold=self.plddt_threshold,
                    device=self.cfg.data.device,
                    max_seq_length=self.max_seq_length,
                    min_seq_length=self.min_seq_length,
                    backbone_atom_mode=self.backbone_atom_mode,
                    max_coil_pct=self.max_coil_pct,
                    min_mean_plddt=self.min_mean_plddt,
                    min_pct_residues_high_plddt=self.min_pct_residues_high_plddt,
                    save_filtered_metadata_path=self.save_filtered_metadata_path,
                )

                total_size = len(full_dataset)
                test_start = int((self.train_split + self.val_split) * total_size)
                test_indices = list(range(test_start, total_size))

                if self.max_test_samples is not None:
                    test_indices = test_indices[:self.max_test_samples]

                self.test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    def _collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        if batch_size == 0:
            return {}

        coords_list = [torch.as_tensor(sample["coords_ca"], dtype=torch.float32) for sample in batch]
        lengths = torch.tensor([coords.shape[0] for coords in coords_list], dtype=torch.long)
        max_len = int(lengths.max().item())

        coords_padded = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for idx, coords in enumerate(coords_list):
            L = coords.shape[0]
            coords_padded[idx, :L] = coords
            sample_mask = batch[idx].get("valid_mask")
            if sample_mask is None:
                mask[idx, :L] = True
            else:
                sample_mask_tensor = torch.as_tensor(sample_mask, dtype=torch.bool)
                mask[idx, : sample_mask_tensor.shape[0]] = sample_mask_tensor

        batch_dict: Dict[str, Any] = {
            "coords_ca": coords_padded,
            "mask": mask,
            "lengths": lengths,
            "names": [sample["name"] for sample in batch],
            "seq": [sample["seq"] for sample in batch],
        }

        if self.backbone_atom_mode.upper() == "FULL":
            for sample in batch:
                if "coords_backbone" not in sample:
                    raise ValueError(
                        "Sample missing 'coords_backbone' while backbone_atom_mode=FULL."
                    )
            backbone_list = [
                torch.as_tensor(sample["coords_backbone"], dtype=torch.float32)
                for sample in batch
            ]
            num_atoms = backbone_list[0].shape[1]
            backbone_padded = torch.zeros(batch_size, max_len, num_atoms, 3, dtype=torch.float32)
            for idx, coords in enumerate(backbone_list):
                L = coords.shape[0]
                backbone_padded[idx, :L] = coords
            batch_dict["coords_backbone"] = backbone_padded

        return batch_dict

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_batch,
        )


