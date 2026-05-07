#!/usr/bin/env python3
"""
__author__: Nabin

CA-CA only inference pipeline:
1) Read ground-truth CA structures.
2) Encode coordinates to tokens and embeddings.
3) Decode tokens back to coordinates.
4) Save tokens, embeddings, and reconstructed CIF with token IDs in B-factor.
Saved in B-factor for easy visualization and loading of tokens for other analysis.
Saved embeddings can be used for other analysis like understandig the embedding space, how things are organized, etc.
5. Compute entropy and perplexity of the tokens.
6. Save the RMSD metrics and entropy/perplexity metrics.


Run:
python inference_test_data.py 
    --checkpoint path/to/checkpoint.ckpt
    --config path/to/config.yaml
    --ground-truth-dir path/to/ground-truth-dir
    --output-dir path/to/output-dir
    --min-seq-len 50 
    --max-seq-len 256 
    --sampler-type rk4 
    --num-steps 100 
    --cfg-scale 3.8 
    --batch-size 200 
    --device cuda 

"""



import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_parser
import biotite.structure.io.pdbx as pdbx_parser


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from structure_tokenizer_model import YetiModel  # noqa: E402
from compute_entropy_perplexity import compute_codebook_stats  # noqa: E402
from utils import kabsch_rmsd  # noqa: E402



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CA-only protein tokenizer inference.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--config", default="yeti_ca_512_8.yaml", help="Model config path.")
    parser.add_argument("--ground-truth-dir", required=True, help="Directory containing CA-only .cif/.pdb files.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--min-seq-len", type=int, default=50, help="Minimum sequence length.")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--sampler-type", default="rk4", choices=["euler", "heun", "rk4"], help="Sampler type.")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of decoder integration steps.")
    parser.add_argument("--cfg-scale", type=float, default=3.8, help="CFG scale.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_structures(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    paths = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in {".cif", ".pdb"}]
    if not paths:
        raise RuntimeError(f"No .cif/.pdb files found in: {directory}")
    return paths


def _res_names_from_ca(structure: struc.AtomArray, ca_mask: np.ndarray) -> np.ndarray:
    """One 3-letter residue name per CA atom, same order as coord[ca_mask]."""
    raw = structure.res_name[ca_mask]
    n = int(raw.shape[0])
    out = np.empty(n, dtype="U3")
    for i in range(n):
        s = str(raw[i]).strip()
        if len(s) >= 3:
            out[i] = s[:3].upper()
        else:
            out[i] = (s + "UNK")[:3].upper()
    return out


def load_ca_coords(
    path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    try:
        if path.suffix.lower() == ".cif":
            structure = pdbx_parser.get_structure(pdbx_parser.CIFFile.read(str(path)), model=1)
        elif path.suffix.lower() == ".pdb":
            structure = pdb_parser.PDBFile.read(str(path)).get_structure(model=1)
        else:
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to read {path.name}: {exc}")
        return None

    ca_mask = structure.atom_name == "CA"
    if not np.any(ca_mask):
        print(f"[WARN] no CA atoms in {path.name}")
        return None

    res_names = _res_names_from_ca(structure, ca_mask)
    res_ids = np.asarray(structure.res_id[ca_mask], dtype=int)
    ins_codes = np.asarray(structure.ins_code[ca_mask])
    chain_ids = np.asarray(structure.chain_id[ca_mask], dtype="U4")
    coords_ca = np.asarray(structure.coord[ca_mask], dtype=np.float32)
    finite_mask = np.isfinite(coords_ca).all(axis=1)
    if not np.any(finite_mask):
        print(f"[WARN] no finite CA coordinates in {path.name}")
        return None
    coords_ca = coords_ca.copy()
    coords_ca[~finite_mask] = 0.0
    return coords_ca, res_names, res_ids, ins_codes, chain_ids, finite_mask


def iter_items(paths: Iterable[Path], min_len: int, max_len: int) -> Iterable[Dict[str, object]]:
    for path in paths:
        loaded = load_ca_coords(path)
        if loaded is None:
            continue
        coords_ca, res_names, res_ids, ins_codes, chain_ids, finite_mask = loaded
        n_res = int(coords_ca.shape[0])
        if n_res < min_len:
            print(f"[SKIP] {path.stem}: Total residues {n_res} < {min_len} (minimum sequence length)")
            continue
        if n_res > max_len:
            print(f"[SKIP] {path.stem}: Total residues {n_res} > {max_len} (maximum sequence length)")
            continue
        yield {
            "protein_id": path.stem,
            "coords_ca": coords_ca,
            "res_names": res_names,
            "res_ids": res_ids,
            "ins_codes": ins_codes,
            "chain_ids": chain_ids,
            "valid_mask": finite_mask,
            "n_res": n_res,
        }


def code_indices_to_scalar(code_indices: torch.Tensor, codebook_size: int, num_codebooks: int) -> torch.Tensor:
    if code_indices.ndim == 2:
        return code_indices
    if num_codebooks > 1:
        # this is not tested, but should work... we used num_codebooks = 1 for all. 
        powers = torch.pow(
            torch.tensor(codebook_size, device=code_indices.device, dtype=torch.long),
            torch.arange(num_codebooks - 1, -1, -1, device=code_indices.device, dtype=torch.long),
        )
        return (code_indices.long() * powers).sum(dim=-1)
    return code_indices.squeeze(-1)


def save_ca_cif(
    coords_ca: np.ndarray,
    tokens: np.ndarray,
    out_path: Path,
    res_names: Optional[np.ndarray] = None,
    res_ids: Optional[np.ndarray] = None,
    ins_codes: Optional[np.ndarray] = None,
    chain_ids: Optional[np.ndarray] = None,
) -> None:
    n_res = int(coords_ca.shape[0])
    arr = struc.AtomArray(n_res)
    arr.coord = coords_ca.astype(np.float32)
    if chain_ids is None:
        arr.chain_id = np.full(n_res, "A", dtype="U4")
    else:
        cid = np.asarray(chain_ids, dtype="U4")
        if cid.shape[0] != n_res:
            raise ValueError(f"chain_ids length {cid.shape[0]} != n_res {n_res}")
        arr.chain_id = cid
    if res_ids is None:
        arr.res_id = np.arange(1, n_res + 1, dtype=int)
    else:
        rid = np.asarray(res_ids, dtype=int)
        if rid.shape[0] != n_res:
            raise ValueError(f"res_ids length {rid.shape[0]} != n_res {n_res}")
        arr.res_id = rid
    if ins_codes is None:
        arr.ins_code = np.full(n_res, "", dtype="U1")
    else:
        ic = np.asarray(ins_codes)
        if ic.shape[0] != n_res:
            raise ValueError(f"ins_codes length {ic.shape[0]} != n_res {n_res}")
        arr.ins_code = np.array([str(x)[:1] if x is not None else "" for x in ic], dtype="U1")
    if res_names is None:
        arr.res_name = np.full(n_res, "GLY", dtype="U3")
    else:
        rn = np.asarray(res_names, dtype="U3")
        if rn.shape[0] != n_res:
            raise ValueError(f"res_names length {rn.shape[0]} != n_res {n_res}")
        arr.res_name = rn
    arr.hetero = np.zeros(n_res, dtype=bool)
    arr.atom_name = np.full(n_res, "CA", dtype="U6")
    arr.element = np.full(n_res, "C", dtype="U2")
    arr.add_annotation("occupancy", dtype=float)
    arr.occupancy[:] = 1.0
    arr.add_annotation("b_factor", dtype=float)
    arr.b_factor[:] = tokens.astype(np.float64)

    cif = pdbx_parser.CIFFile()
    pdbx_parser.set_structure(cif, arr)
    cif.write(str(out_path))


RMSD_FIELDS = ["protein_id", "n_residues", "n_valid_ca", "kabsch_rmsd", "status"]
CODEBOOK_STATS_FIELDS = [
    "entropy",
    "perplexity",
    "usage_fraction",
    "num_active_tokens",
    "total_tokens",
]


def process_batch(
    items: List[Dict[str, object]],
    *,
    model: YetiModel,
    device: torch.device,
    num_steps: int,
    sampler_type: str,
    cfg_scale: Optional[float],
    codebook_size: int,
    num_codebooks: int,
    tokens_dir: Path,
    pre_quant_dir: Path,
    post_quant_dir: Path,
    recon_dir: Path,
    rmsd_writer: csv.DictWriter,
    token_sequences: List[np.ndarray],
) -> None:
    if not items:
        return

    batch_size = len(items)
    max_len = max(int(item["n_res"]) for item in items)
    coords_b = torch.zeros((batch_size, max_len, 3), dtype=torch.float32, device=device)
    mask_b = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for i, item in enumerate(items):
        n_res = int(item["n_res"])
        coords = torch.from_numpy(item["coords_ca"]).to(device=device, dtype=torch.float32)
        valid_mask = torch.from_numpy(item["valid_mask"]).to(device=device, dtype=torch.bool)
        coords_b[i, :n_res] = coords
        mask_b[i, :n_res] = valid_mask

    encoder_out = model.quantized_encoder(coords_b, mask_b)
    code_indices = encoder_out["code_indices"]
    centers = encoder_out["centers"]
    scalar_tokens = code_indices_to_scalar(code_indices, codebook_size, num_codebooks)
    pre_quant = encoder_out["encoder_latents"]
    post_quant = encoder_out["token_embeddings"]
    

    # print(encoder_out)
    total_rmsd = 0.0
    total_valid_proteins = 0.0

    recon = model.sample_from_tokens(
        code_indices=code_indices,
        mask=mask_b,
        num_steps=num_steps,
        centers=centers,
        sampler_type=sampler_type,
        cfg_scale=cfg_scale,
    )
    if recon.ndim == 4:
        recon = recon[:, :, 1, :]

    for i, item in enumerate(items):
        protein_id = str(item["protein_id"])
        n_res = int(item["n_res"])
        valid_mask = np.asarray(item["valid_mask"], dtype=bool)

        token_row = scalar_tokens[i]
        if int(token_row.shape[0]) < n_res:
            print(f"[SKIP] {protein_id}: token length mismatch")
            continue
        tokens = token_row[:n_res].detach().cpu().numpy().astype(np.int64)
        if tokens.shape[0] != n_res:
            print(f"[SKIP] {protein_id}: token length mismatch")
            continue
        token_sequences.append(tokens[valid_mask].copy())

        pre = pre_quant[i, :n_res].detach().cpu().numpy()
        post = post_quant[i, :n_res].detach().cpu().numpy()
        recon_ca = recon[i, :n_res].detach().cpu().numpy().astype(np.float32)
        if recon_ca.shape[0] != n_res:
            print(f"[SKIP] {protein_id}: reconstruction length mismatch")
            continue

        coords_ca_true = np.asarray(item["coords_ca"], dtype=np.float64)
        rmsd_val = ""
        status = "ok"
        n_valid = int(valid_mask.sum())
        if n_valid > 0:
            rmsd_val = round(float(kabsch_rmsd(recon_ca[valid_mask], coords_ca_true[valid_mask])), 3)
            print(f"  {protein_id}: n_res={n_res} kabsch_rmsd={rmsd_val}")
            total_rmsd += rmsd_val
            total_valid_proteins += 1
        else:
            status = "no_valid_ca"
            print(f"  {protein_id}: n_res={n_res} no valid CA for RMSD")

        tokens[~valid_mask] = -1
        np.savetxt(tokens_dir / f"{protein_id}_tokens.txt", tokens, fmt="%d")
        np.savetxt(pre_quant_dir / f"{protein_id}_encoder_pre_quant.txt", pre, fmt="%.7f")
        np.savetxt(post_quant_dir / f"{protein_id}_encoder_post_quant.txt", post, fmt="%.7f")
        save_ca_cif(
            recon_ca,
            tokens,
            recon_dir / f"{protein_id}_recon.cif",
            res_names=np.asarray(item["res_names"], dtype="U3"),
            res_ids=np.asarray(item["res_ids"], dtype=int),
            ins_codes=np.asarray(item["ins_codes"]),
            chain_ids=np.asarray(item["chain_ids"], dtype="U4"),
        )

        rmsd_writer.writerow({
            "protein_id": protein_id,
            "n_residues": n_res,
            "n_valid_ca": n_valid,
            "kabsch_rmsd": rmsd_val,
            "status": status,
        })
    
    print(f"Mean RMSD: {round(total_rmsd/total_valid_proteins, 3)}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(THIS_DIR) / cfg_path
    cfg = OmegaConf.load(str(cfg_path))

    output_dir = Path(args.output_dir)
    tokens_dir = output_dir / "tokens"
    pre_quant_dir = output_dir / "embeddings_pre_quant"
    post_quant_dir = output_dir / "embeddings_post_quant"
    recon_dir = output_dir / "recon_coords"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    pre_quant_dir.mkdir(parents=True, exist_ok=True)
    post_quant_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    min_len = int(args.min_seq_len)
    max_len = int(args.max_seq_len)
    if min_len > max_len:
        raise ValueError(f"Invalid length range: {min_len} > {max_len}")

    num_steps = int(
        args.num_steps
        if args.num_steps is not None
        else getattr(getattr(cfg, "train", {}), "rmsd_num_steps", 50) or 50
    )
    device = get_device(args.device)
    model = YetiModel.load_from_checkpoint(args.checkpoint, cfg=cfg, map_location=device, strict=True)
    model.eval()
    model.to(device)

    quant_cfg = getattr(cfg.model, "quantizer", None)
    codebook_size = int(getattr(quant_cfg, "codebook_size", 8192) or 8192)
    num_codebooks = int(getattr(quant_cfg, "num_codebooks", 1) or 1)

    paths = discover_structures(Path(args.ground_truth_dir))
    print(f"#" * 25)
    print(f"[INFO] structures={len(paths)} batch_size={args.batch_size} device={device} seed={args.seed} min_len={min_len} max_len={max_len} num_steps={num_steps} sampler_type={args.sampler_type}")
    print(f"#" * 25)

    rmsd_path = output_dir / "rmsd_metrics.csv"
    codebook_stats_path = output_dir / "entropy_perplexity_metrics.csv"
    token_sequences: List[np.ndarray] = []
    with rmsd_path.open("w", newline="") as rmsd_file, torch.no_grad():
        rmsd_writer = csv.DictWriter(rmsd_file, fieldnames=RMSD_FIELDS)
        rmsd_writer.writeheader()

        batch: List[Dict[str, object]] = []
        for item in iter_items(paths, min_len=min_len, max_len=max_len):
            batch.append(item)
            if len(batch) >= args.batch_size:
                process_batch(
                    batch,
                    model=model,
                    device=device,
                    num_steps=num_steps,
                    sampler_type=args.sampler_type,
                    cfg_scale=args.cfg_scale,
                    codebook_size=codebook_size,
                    num_codebooks=num_codebooks,
                    tokens_dir=tokens_dir,
                    pre_quant_dir=pre_quant_dir,
                    post_quant_dir=post_quant_dir,
                    recon_dir=recon_dir,
                    rmsd_writer=rmsd_writer,
                    token_sequences=token_sequences,
                )
                batch.clear()
        if batch:
            process_batch(
                batch,
                model=model,
                device=device,
                num_steps=num_steps,
                sampler_type=args.sampler_type,
                cfg_scale=args.cfg_scale,
                codebook_size=codebook_size,
                num_codebooks=num_codebooks,
                tokens_dir=tokens_dir,
                pre_quant_dir=pre_quant_dir,
                post_quant_dir=post_quant_dir,
                recon_dir=recon_dir,
                rmsd_writer=rmsd_writer,
                token_sequences=token_sequences,
            )

    if token_sequences:
        codebook_stats = compute_codebook_stats(token_sequences, vocab_size=codebook_size)
        print(codebook_stats)
        with codebook_stats_path.open("w", newline="") as stats_file:
            stats_writer = csv.DictWriter(stats_file, fieldnames=CODEBOOK_STATS_FIELDS)
            stats_writer.writeheader()
            stats_writer.writerow(codebook_stats)

        print(
            "[INFO] codebook_stats "
            f"entropy={codebook_stats['entropy']:.6f} "
            f"perplexity={codebook_stats['perplexity']:.6f} "
            f"usage_fraction={codebook_stats['usage_fraction']:.6f} "
            f"num_active_tokens={codebook_stats['num_active_tokens']} "
            f"total_tokens={codebook_stats['total_tokens']}"
        )
        print(f"[INFO] entropy_perplexity_csv={codebook_stats_path}")
    else:
        print("[WARN] no valid token sequences found; skipped entropy/perplexity computation")

    print(f"[INFO] rmsd_csv={rmsd_path}")
    print(f"[INFO] done: {output_dir}")


if __name__ == "__main__":
    main()
