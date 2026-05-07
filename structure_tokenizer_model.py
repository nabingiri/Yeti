"""
__author__: Nabin
Structure tokenizer model components.
Encoder uses transformer blocks; decoder uses StripedHyena blocks.
StripedHyena blocks are from the stripedhyena repository.
https://github.com/togethercomputer/stripedhyena

We use the default configuration from the stripedhyena repository.
"""

from typing import Dict, Optional, Any
import math
import os
import sys

import torch
import torch.distributed as dist
from torch import nn
import pytorch_lightning as pl

# stripedhyena-main contains the stripedhyena package
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STRIPEDHYENA_ROOT = os.path.join(_THIS_DIR, "stripedhyena-main")
if _STRIPEDHYENA_ROOT not in sys.path:
    sys.path.insert(0, _STRIPEDHYENA_ROOT)

from transformer_blocks import TransformerEncoderBlock, sinusoidal_time_embedding
from lookup_free_quantization import LFQ
import logging

logger = logging.getLogger(__name__)

# DotDict for StripedHyena config (they use config.attr and config.get())
class _DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# StripedHyena imports (after path is set)
try:
    from stripedhyena.model import get_block as _sh_get_block
except ImportError as e:
    raise ImportError(
        f"StripedHyena not found at {_STRIPEDHYENA_ROOT}. "
        f"Ensure stripedhyena-main is extracted there. Original: {e}"
    ) from e


def _make_hyena_config(
    hidden_dim: int,
    depth: int,
    num_heads: int,
    state_size: int = 2,
    short_filter_length: int = 3,
    num_filters: Optional[int] = None,
    short_filter_bias: bool = True,
    **kwargs: Any,
) -> Any:
    """Build a StripedHyena-style config for Hyena-only blocks."""
    if num_filters is None:
        num_filters = hidden_dim
    assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
    assert hidden_dim % num_filters == 0 and num_filters <= hidden_dim
    cfg = {
        "hidden_size": hidden_dim,
        "num_attention_heads": num_heads,
        "num_filters": num_filters,
        "state_size": state_size,
        "short_filter_length": short_filter_length,
        "short_filter_bias": short_filter_bias,
        "attn_layer_idxs": [],  # Hyena-only
        "hyena_layer_idxs": list(range(depth)),
        "eps": 1e-5,
        "params_dtype": torch.float32,
        "model_parallel_size": 1,
        "use_flash_attn": False,
        "use_flash_rmsnorm": False,
        "use_flashfft": False,
        "use_flash_depthwise": False,
        "column_split_hyena": True,
        "hyena_block_dtype": torch.float32,
        "mlp_dtype": torch.float32,  # keep same as block dtype so optimizer gets single dtype (bf16-mixed still casts activations)
    }
    cfg.update(kwargs)
    return _DotDict(cfg)


class StructureEncoder(nn.Module):
    """
    Encodes mean-centered CA coordinates into latent representations suitable for LFQ.
    Uses transformer blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        use_rotary: bool = True,
        coord_mode: str = "CA",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.coord_mode = (coord_mode or "CA").upper()
        if self.coord_mode not in {"CA", "FULL"}:
            raise ValueError(f"Unsupported coord_mode={coord_mode}. Expected 'CA' or 'FULL'.")
        self.coord_dim = 12 if self.coord_mode == "FULL" else 3

        self.input_proj = nn.Linear(self.coord_dim, hidden_dim)
        self.positional_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rotary=use_rotary,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            coords: (B, L, D) padded coordinates where D=3 for CA mode, D=12 for FULL mode.
            mask: (B, L) boolean mask indicating valid residues.
        Returns:
            dict with latents, centered coordinates, and per-sample centers.
        """
        if coords.ndim != 3 or coords.shape[-1] != self.coord_dim:
            raise ValueError(
                f"coords must have shape (B, L, {self.coord_dim}); got {coords.shape}"
            )
        if mask.dtype != torch.bool:
            mask = mask.bool()

        mask_float = mask.unsqueeze(-1).to(dtype=coords.dtype)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        # mean center the coordinates
        centers = (coords * mask_float).sum(dim=1, keepdim=True) / lengths.unsqueeze(-1)
        centered = (coords - centers) * mask_float
        x = self.input_proj(centered)

        seq_len = coords.shape[1]
        pos_ids = torch.arange(seq_len, device=coords.device)
        pos_ids = torch.clamp(pos_ids, max=self.max_seq_len - 1)
        pos_emb = self.positional_emb(pos_ids).unsqueeze(0)
        x = x + pos_emb
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        latents = self.final_norm(x)
        return {
            "latents": latents,
            "centered_coords": centered,
            "centers": centers.squeeze(1),
            "mask": mask,
        }


class MLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        # If no hidden dim specified, keep the widest dimension to avoid issues.
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Stabilizes variance for quantization
            nn.GELU(),                 
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)




class QuantizedStructureEncoder(nn.Module):
    """
    Adds lookup-free quantization on top of the StructureEncoder and produces token embeddings.
    """

    def __init__(
        self,
        encoder_cfg: Dict,
        quantizer_cfg: Dict,
        decoder_cfg: Dict,
    ):
        super().__init__()
        self.encoder = StructureEncoder(**encoder_cfg)
        self.hidden_dim_encoder = self.encoder.hidden_dim
        
        self.hidden_dim_decoder = decoder_cfg["hidden_dim"]

        codebook_size = quantizer_cfg["codebook_size"]
        num_codebooks = quantizer_cfg["num_codebooks"]
        post_quant_proj_dim = quantizer_cfg["post_quant_proj_dim"]
        codebook_dim = int(math.log2(codebook_size))
        if codebook_dim * num_codebooks <= 0:
            raise ValueError("Invalid LFQ configuration.")

        self.lfq_dim = codebook_dim * num_codebooks

        self.pre_quant_proj = MLPProjector(
            input_dim=self.hidden_dim_encoder,
            output_dim=self.lfq_dim,
            hidden_dim=self.hidden_dim_encoder,
        )
        self.post_quant_proj = MLPProjector(
            input_dim=self.lfq_dim,
            output_dim=post_quant_proj_dim,
            hidden_dim=post_quant_proj_dim,
        )


        self.quantizer = LFQ(
            dim=self.lfq_dim,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            entropy_loss_weight=quantizer_cfg.get("entropy_loss_weight", 0.1),
            commitment_loss_weight=quantizer_cfg.get("commitment_loss_weight", 0.25),
        )

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder(coords, mask)
        lfq_input = self.pre_quant_proj(enc_out["latents"])

        ret, breakdown = self.quantizer(
            lfq_input,
            mask=None,
            return_loss_breakdown=True,
        )
        quantized = ret.quantized
        indices = ret.indices
        quant_loss = ret.entropy_aux_loss
        
        token_embeddings = self.post_quant_proj(quantized)

        enc_out.update(
            {
                "lfq_loss": quant_loss,
                "code_indices": indices,
                "quantized_latents": quantized,
                "token_embeddings": token_embeddings,
                "encoder_latents": enc_out["latents"],
                "lfq_per_sample_entropy": breakdown.per_sample_entropy,
                "lfq_codebook_entropy": breakdown.batch_entropy,  
                "lfq_commit_loss": breakdown.commitment,
            }
        )
        return enc_out

    def indices_to_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        # LFQ uses indices_to_codes 
        decoded = self.quantizer.indices_to_codes(indices, project_out=False)
        return self.post_quant_proj(decoded)


class StructureDecoder(nn.Module):
    """
    Predicts the velocity field v_t given noisy coordinates x_t, time t, and discrete token context.
    Uses StripedHyena Hyena blocks; coords, tokens, and time are fused into one sequence, no cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        token_dim: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        coord_dim: int = 3,
        state_size: int = 2,
        short_filter_length: int = 3,
        num_filters: Optional[int] = None,
        short_filter_bias: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.time_embed_dim = hidden_dim
        self.coord_dim = coord_dim
        self.token_dim = token_dim

        self.input_proj = nn.Linear(self.coord_dim, hidden_dim)
        self.positional_emb = nn.Embedding(max_seq_len, hidden_dim)
        if token_dim == hidden_dim:
            self.context_in = nn.Identity()
        else:
            self.context_in = nn.Linear(token_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        sh_config = _make_hyena_config(
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            state_size=state_size,
            short_filter_length=short_filter_length,
            num_filters=num_filters,
            short_filter_bias=short_filter_bias,
        )
        self.blocks = nn.ModuleList(
            [_sh_get_block(sh_config, layer_idx, flash_fft=None) for layer_idx in range(depth)]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.coord_dim)

    def forward(
        self,
        noisy_coords: torch.Tensor,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if token_mask.dtype != torch.bool:
            token_mask = token_mask.bool()

        seq_len = noisy_coords.shape[1]
        pos_ids = torch.arange(seq_len, device=noisy_coords.device)
        pos_ids = torch.clamp(pos_ids, max=self.max_seq_len - 1)
        pos_emb = self.positional_emb(pos_ids).unsqueeze(0)

        if noisy_coords.ndim != 3 or noisy_coords.shape[-1] != self.coord_dim:
            raise ValueError(
                f"noisy_coords must have shape (B, L, {self.coord_dim}); got {noisy_coords.shape}"
            )
        x_coord = self.input_proj(noisy_coords) + pos_emb
        x_coord = self.dropout(x_coord)

        if tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"tokens last dim must be {self.token_dim} (token_dim) but got {tokens.shape[-1]}"
            )
        x_tok = self.context_norm(self.context_in(tokens))
        x_tok = x_tok * token_mask.unsqueeze(-1).to(dtype=x_tok.dtype)

        t_emb = sinusoidal_time_embedding(timesteps, self.time_embed_dim)
        cond = self.time_mlp(t_emb)  # (B, H)

        # Fuse coords + tokens + time into one sequence for Hyena blocks
        x = self.input_norm(x_coord + x_tok + cond.unsqueeze(1))
        mask_float = mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * mask_float

        padding_mask = mask.float().to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x, _ = block(x, inference_params=None, padding_mask=padding_mask)

        v_pred = self.output_proj(self.final_norm(x))
        v_pred = v_pred * mask_float
        return v_pred


class StructureTokenizerFlowModel(pl.LightningModule):
    """
    Example configurations: (we ran only CA to CA). Would be nice to see how other configurations perform. Fingers crossed! Need more time to see how other configurations perform.
    - CA to CA: encoder_coord_mode=CA, decoder_coord_mode=CA
    - FULL to FULL: encoder_coord_mode=FULL, decoder_coord_mode=FULL
    - CA to FULL: encoder_coord_mode=CA, decoder_coord_mode=FULL
    - FULL to CA: encoder_coord_mode=FULL, decoder_coord_mode=CA
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.backbone_atom_count = 4 # number of atoms in the backbone (N, CA, C, O)

        # Resolve encoder and decoder coord modes separately, this is for the example configurations.
        data_mode = getattr(cfg.data, "backbone_atom_mode", "CA")
        encoder_mode_cfg = getattr(cfg.model, "encoder_coord_mode", "auto")
        decoder_mode_cfg = getattr(cfg.model, "decoder_coord_mode", "auto")
        
        
        self.encoder_coord_mode = self._resolve_single_coord_mode(data_mode, encoder_mode_cfg, "encoder")
        self.decoder_coord_mode = self._resolve_single_coord_mode(data_mode, decoder_mode_cfg, "decoder")
        
        self.encoder_coord_dim = 12 if self.encoder_coord_mode == "FULL" else 3 # 12 for FULL mode, 3 for CA mode. xyz coordinates for each atom.
        self.decoder_coord_dim = 12 if self.decoder_coord_mode == "FULL" else 3
        
        self.coord_mode = self.decoder_coord_mode
        self.coord_dim = self.decoder_coord_dim
        
        logger.info(f"Model coord modes: encoder={self.encoder_coord_mode} ({self.encoder_coord_dim}D), "
                    f"decoder={self.decoder_coord_mode} ({self.decoder_coord_dim}D)")

        model_cfg = cfg.model
        encoder_cfg = {
            "hidden_dim": model_cfg.encoder.hidden_dim,
            "depth": model_cfg.encoder.depth,
            "num_heads": model_cfg.encoder.num_heads,
            "max_seq_len": getattr(model_cfg.encoder, "max_seq_len", 1024),
            "dropout": getattr(model_cfg.encoder, "dropout", 0.1),
            "use_rotary": getattr(model_cfg.encoder, "use_rotary", True),
            "coord_mode": self.encoder_coord_mode,
        }
        quantizer_cfg = {
            "codebook_size": model_cfg.quantizer.codebook_size,
            "num_codebooks": model_cfg.quantizer.num_codebooks, 
            "post_quant_proj_dim": model_cfg.quantizer.post_quant_proj_dim,
            "entropy_loss_weight": getattr(model_cfg.quantizer, "entropy_loss_weight", 0.1),
            "commitment_loss_weight": getattr(
                model_cfg.quantizer, "commitment_loss_weight", 0.25
            )
        }
        decoder_cfg = {
            "hidden_dim": model_cfg.decoder.hidden_dim,
            "depth": model_cfg.decoder.depth,
            "num_heads": model_cfg.decoder.num_heads,
            "token_dim": quantizer_cfg["post_quant_proj_dim"],
            "max_seq_len": getattr(model_cfg.decoder, "max_seq_len", encoder_cfg["max_seq_len"]),
            "dropout": getattr(model_cfg.decoder, "dropout", 0.1),
            "coord_dim": self.decoder_coord_dim,
            "state_size": getattr(model_cfg.decoder, "state_size", 2),
            "short_filter_length": getattr(model_cfg.decoder, "short_filter_length", 3),
            "num_filters": getattr(model_cfg.decoder, "num_filters", None),
            "short_filter_bias": getattr(model_cfg.decoder, "short_filter_bias", True),
        }

        self.quantized_encoder = QuantizedStructureEncoder(encoder_cfg, quantizer_cfg, decoder_cfg)
        self.decoder = StructureDecoder(**decoder_cfg)

        self.lambda_flow = getattr(cfg.loss, "lambda_flow", 1.0)
        self.lambda_lfq = getattr(cfg.loss, "lambda_lfq", 1.0)
        
        self.coord_scale = float(getattr(model_cfg, "coord_scale", 0.1) or 0.1)
        if self.coord_scale <= 0:
            raise ValueError("model.coord_scale must be positive.")
        self.inv_coord_scale = 1.0 / self.coord_scale # inverse of the coordinate scale.

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = (pred - target) ** 2
        mask_f = mask.unsqueeze(-1).to(dtype=pred.dtype)
        denom = mask.sum().clamp(min=1).to(dtype=pred.dtype)
        return (diff * mask_f).sum() / denom

    @staticmethod
    def _resolve_single_coord_mode(data_mode: str, model_mode: str, role: str = "model") -> str:
        """
        
        Args:
            data_mode: The backbone_atom_mode from data config ("CA" or "FULL")
            model_mode: The coord_mode from model config ("CA", "FULL", or "auto")
        
        Returns:
            Resolved coord mode ("CA" or "FULL")
        """
        data_mode_norm = str(data_mode or "CA").upper()
        model_mode_norm = str(model_mode or "AUTO").upper()
        
        if model_mode_norm == "AUTO":
            coord_mode = data_mode_norm
        else:
            coord_mode = model_mode_norm
            
        if coord_mode not in {"CA", "FULL"}:
            raise ValueError(f"{role}_coord_mode must be 'CA', 'FULL', or 'auto'. Got {coord_mode}.")
        
        # If model wants FULL but data only provides CA, that's an error
        if coord_mode == "FULL" and data_mode_norm != "FULL":
            raise ValueError(
                f"{role}_coord_mode='FULL' requires data.backbone_atom_mode='FULL' "
                f"(currently data provides '{data_mode_norm}')."
            )
        return coord_mode


    def _flatten_backbone(self, coords_backbone: torch.Tensor) -> torch.Tensor:
        if coords_backbone.ndim != 4 or coords_backbone.shape[2] != self.backbone_atom_count:
            raise ValueError(
                "Expected coords_backbone with shape (B, L, 4, 3) when using FULL mode."
            )
        return coords_backbone.reshape(coords_backbone.shape[0], coords_backbone.shape[1], -1)

    def _unflatten_backbone(self, coords_flat: torch.Tensor) -> torch.Tensor:
        if coords_flat.shape[-1] != self.coord_dim:
            raise ValueError(
                f"Cannot unflatten tensor with last dim {coords_flat.shape[-1]} (expected {self.coord_dim})."
            )
        return coords_flat.view(coords_flat.shape[0], coords_flat.shape[1], self.backbone_atom_count, 3)

    def _get_encoder_coords_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get coordinates for encoder input based on encoder_coord_mode."""
        if self.encoder_coord_mode == "FULL":
            coords_backbone = batch.get("coords_backbone")
            if coords_backbone is None:
                raise ValueError("Batch is missing 'coords_backbone' required for encoder FULL mode.")
            return self._flatten_backbone(coords_backbone)
        return batch["coords_ca"]
    
    def _get_decoder_coords_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get coordinates for decoder target based on decoder_coord_mode."""
        if self.decoder_coord_mode == "FULL":
            coords_backbone = batch.get("coords_backbone")
            if coords_backbone is None:
                raise ValueError("Batch is missing 'coords_backbone' required for decoder FULL mode.")
            return self._flatten_backbone(coords_backbone)
        return batch["coords_ca"]

    def _sample_base(self, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sample centered Gaussian noise base for flow matching."""
        # Keep everything in target dtype (ran into issues earlier, so lets keep everything to same dtype).
        mask_f = mask.unsqueeze(-1).to(dtype=target.dtype)  # (B, L, 1)

        # Sample noise and zero out masked positions.
        # Standard normal distribution with mean 0 and variance 1.
        noise = torch.randn_like(target) * mask_f
        
        # Center the noise (subtract mean across valid positions)
        valid_count = mask.unsqueeze(-1).sum(dim=1, keepdim=True).clamp(min=1)  # int64
        mean = noise.sum(dim=1, keepdim=True) / valid_count.to(dtype=noise.dtype)
        noise = (noise - mean) * mask_f
        
        return noise

    def _compute_ca_centroid(self, batch: Dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """
        Compute CA centroid from batch. Always uses coords_ca for consistency and we must have CA coordinates in the batch.
        
        Returns:
            centers: (B, 3) CA centroid for each sample
        """
        coords_ca = batch["coords_ca"].to(self.device)
        mask_float = mask.unsqueeze(-1).to(dtype=coords_ca.dtype)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        centers = (coords_ca * mask_float).sum(dim=1) / lengths.to(dtype=coords_ca.dtype)
        return centers  # (B, 3)

    def _center_coords(
        self, 
        coords: torch.Tensor, 
        centers: torch.Tensor, 
        mask: torch.Tensor,
        coord_dim: int
    ) -> torch.Tensor:
        """
        Center coordinates using provided centers (CA centroid).
        
        For CA coords (dim=3): subtract centers directly
        For FULL coords (dim=12): broadcast CA centroid to all 4 atoms
        
        Args:
            coords: (B, L, coord_dim) coordinates to center
            centers: (B, 3) CA centroid
            mask: (B, L) boolean mask
            coord_dim: 3 for CA, 12 for FULL
            
        Returns:
            centered: (B, L, coord_dim) centered coordinates
        """
        mask_float = mask.unsqueeze(-1).to(dtype=coords.dtype)
        
        if coord_dim == 3:
            # CA mode: direct subtraction
            centered = (coords - centers.unsqueeze(1)) * mask_float
        else:
            # FULL mode: broadcast CA centroid to all 4 atoms
            # centers: (B, 3) → (B, 1, 1, 3) → (B, 1, 4, 3) → (B, 1, 12)
            # Haven't really tested.... but it should work.
            centers_expanded = centers.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.backbone_atom_count, -1)
            centers_flat = centers_expanded.reshape(centers.shape[0], 1, coord_dim)
            centered = (coords - centers_flat) * mask_float
            
        return centered



    def _log_lfq_metrics(self, encoder_out, stage: str, on_step: bool, batch_size: int):
        """Log LFQ quantizer metrics."""
        if encoder_out is None:
            return

        commit = encoder_out.get("lfq_commit_loss")
        codebook_entropy = encoder_out.get("lfq_codebook_entropy")
        per_sample_entropy = encoder_out.get("lfq_per_sample_entropy")

        if commit is None or codebook_entropy is None or per_sample_entropy is None:
            logger.warning(f"LFQ metrics encountered None in encoder output during {stage} stage.")
            return

        self.log(f"{stage}_lfq_commit_loss", commit, on_step=on_step, on_epoch=True, sync_dist=False, batch_size=batch_size)
        self.log(f"{stage}_lfq_codebook_entropy", codebook_entropy, on_step=on_step, on_epoch=True, sync_dist=False, batch_size=batch_size)

        # Perplexity is exp(entropy) because entropy is computed with natural logs
        codebook_perplexity = torch.exp(codebook_entropy)
        self.log(f"{stage}_lfq_codebook_perplexity", codebook_perplexity, on_step=on_step, on_epoch=True, sync_dist=False, batch_size=batch_size)

        codebook_perplexity_percentage = codebook_perplexity / self.cfg.model.quantizer.codebook_size
        self.log(f"{stage}_lfq_codebook_perplexity_percentage", codebook_perplexity_percentage, on_step=on_step, on_epoch=True, sync_dist=False, batch_size=batch_size)
        self.log(f"{stage}_lfq_sample_entropy", per_sample_entropy, on_step=on_step, on_epoch=True, sync_dist=False, batch_size=batch_size)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        """Shared step for validation and test"""
        mask = batch["mask"].to(self.device)
        
        # Get encoder input and decoder target coords (may be different for asymmetric modes)
        encoder_coords = self._get_encoder_coords_from_batch(batch).to(self.device)
        decoder_coords = self._get_decoder_coords_from_batch(batch).to(self.device)
        
        # Compute CA centroid (always from CA coords for consistency)
        ca_centers = self._compute_ca_centroid(batch, mask)  # (B, 3)

        # Run encoder on encoder coords
        encoder_out = self.quantized_encoder(encoder_coords, mask)
        mask = encoder_out["mask"]
        tokens = encoder_out["token_embeddings"]
        
        # Center decoder target using CA centroid
        target_centered = self._center_coords(decoder_coords, ca_centers, mask, self.decoder_coord_dim)
        target = target_centered * self.coord_scale # scale we are using is 0.0667

        # Sample noise base (in decoder coord space)
        base = self._sample_base(target, mask)

        
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        t_exp = t.view(-1, 1, 1)

        # Interpolate linearly between base and target
        xt = (1.0 - t_exp) * base + t_exp * target

        # Get the velocity for a given pair of noise and target points.
        ut = target - base
        
        # Apply mask
        mask_float = mask.unsqueeze(-1).to(dtype=xt.dtype)
        xt = xt * mask_float
        ut = ut * mask_float

        # Predict velocity
        v_pred = self.decoder(xt, tokens, t, mask, mask)
        
        # Loss calculation
        flow_loss = self._masked_mse(v_pred, ut, mask)
        total_loss = (
            self.lambda_flow * flow_loss
            + self.lambda_lfq * encoder_out["lfq_loss"]
        )

        bs = encoder_coords.shape[0]
        self.log(f"{stage}_flow_loss", flow_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(f"{stage}_lfq_loss", encoder_out["lfq_loss"], on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(
            f"{stage}_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=bs,
        )
        self._log_lfq_metrics(encoder_out, stage, on_step=True, batch_size=bs)

        return total_loss

    def training_step(self, batch, batch_idx):
        mask = batch["mask"].to(self.device)
        
        # Get encoder input and decoder target coords (may be different for asymmetric modes)
        encoder_coords = self._get_encoder_coords_from_batch(batch).to(self.device)
        decoder_coords = self._get_decoder_coords_from_batch(batch).to(self.device)
        
        # Compute CA centroid (always from CA coords for consistency)
        ca_centers = self._compute_ca_centroid(batch, mask)  # (B, 3)

        # Run encoder on encoder coords
        encoder_out = self.quantized_encoder(encoder_coords, mask)
        mask = encoder_out["mask"]
        tokens = encoder_out["token_embeddings"]
        
        # Center decoder target using CA centroid
        # For symmetric modes, this should match encoder's centering
        # For asymmetric modes, this ensures consistent centering
        target_centered = self._center_coords(decoder_coords, ca_centers, mask, self.decoder_coord_dim)
        target = target_centered * self.coord_scale

        # Sample noise base 
        base = self._sample_base(target, mask)

        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        t_exp = t.view(-1, 1, 1)
        xt = (1.0 - t_exp) * base + t_exp * target
        ut = target - base
        
        # Apply mask
        mask_float = mask.unsqueeze(-1).to(dtype=xt.dtype)
        xt = xt * mask_float
        ut = ut * mask_float

        # Classifier-free guidance dropout
        if self.training:
            dropout_prob = float(getattr(self.cfg.model.decoder, "cond_dropout", 0.0))
            if dropout_prob > 0 and torch.rand(1).item() < dropout_prob:
                tokens = torch.zeros_like(tokens)

        # Predict velocity
        v_pred = self.decoder(xt, tokens, t, mask, mask)
        
        # Loss calculation
        flow_loss = self._masked_mse(v_pred, ut, mask)
        bs = encoder_coords.shape[0]
        total_loss = self.lambda_flow * flow_loss + self.lambda_lfq * encoder_out["lfq_loss"]

        # Log metrics
        self.log("train_flow_loss", flow_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train_lfq_loss", encoder_out["lfq_loss"], on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self._log_lfq_metrics(encoder_out, "train", on_step=True, batch_size=bs)

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = self._shared_step(batch, stage="val")
        
        return total_loss

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")



    def configure_optimizers(self):
        lr = float(self.hparams.cfg.optimizer.lr)
        optimizer = torch.optim.AdamW(lr=lr, betas=(0.9, 0.98))

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        configured_warmup = getattr(self.hparams.cfg.optimizer, "warmup_steps", None)
        if configured_warmup is None:
            warmup_steps = min(2000, int(0.05 * total_steps))
        else:
            warmup_steps = max(0, min(int(configured_warmup), total_steps))

        constant_frac = float(getattr(self.hparams.cfg.optimizer, "constant_lr_fraction", 0.3))
        constant_frac = min(max(constant_frac, 0.0), 1.0)
        constant_end = int(constant_frac * total_steps)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < constant_end:
                return 1.0
            progress = float(current_step - constant_end) / float(
                max(1, total_steps - constant_end)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def encode_tokens(self, coords: torch.Tensor, mask: torch.Tensor):
        """
        Utility to obtain discrete tokens from raw coordinates.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.quantized_encoder(coords, mask)
        return outputs["code_indices"], outputs["centers"]

    def tokens_to_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        return self.quantized_encoder.indices_to_embeddings(indices)

    def sample_from_tokens(
        self,
        code_indices: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 50,
        centers: torch.Tensor = None,
        sampler_type: str = "rk4",
        cfg_scale: float = None,
    ) -> torch.Tensor:
        """
        Reconstruct coordinates from discrete tokens via flow integration.

        cfg_scale: If not None and != 1.0, use classifier-free guidance
        and unconditioned velocity as v = v_uncond + cfg_scale * (v_cond - v_uncond).
        cfg_scale=1.0 or None = no guidance (single pass). cfg_scale>1 strengthens conditioning.
        """
        mask = mask.to(code_indices.device).bool()
        tokens = self.tokens_to_embeddings(code_indices)
        return self._sample_with_conditioning(tokens, mask, num_steps, centers, sampler_type, cfg_scale=cfg_scale)

    def _sample_with_conditioning(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 50,
        centers: torch.Tensor = None,
        sampler_type: str = "rk4",
        cfg_scale: float = None,
    ) -> torch.Tensor:
        """
        Core sampling logic used by sample_from_tokens.
        Integrates the flow ODE from noise to structure using the provided conditioning.

        cfg_scale: If not None and != 1.0, classifier-free guidance is applied:
        v = v_uncond + cfg_scale * (v_cond - v_uncond). Requires two decoder passes per step.
        """
        device = tokens.device
        mask = mask.to(device).bool()
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]
        use_cfg = cfg_scale is not None and cfg_scale != 1.0

        # Run decoder in float32 during inference to avoid dtype mismatch with mixed-precision blocks. Ran into issues earlier.
        self.decoder.float()

        base = torch.zeros(
            batch_size,
            seq_len,
            self.decoder_coord_dim,
            device=device,
            dtype=tokens.dtype,
        )
        x = self._sample_base(base, mask)
        dt = 1.0 / max(num_steps, 1)

        def get_velocity(x_t: torch.Tensor, t_val: float) -> torch.Tensor:
            t_vec = torch.full((batch_size,), t_val, device=device, dtype=x_t.dtype)
            if use_cfg:
                v_cond = self.decoder(x_t, tokens, t_vec, mask, mask)
                v_uncond = self.decoder(x_t, torch.zeros_like(tokens), t_vec, mask, mask) # null tokens
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            return self.decoder(x_t, tokens, t_vec, mask, mask)

        for step in range(num_steps):
            t = step / num_steps
            
            if sampler_type == "euler":
                v = get_velocity(x, t)
                x = x + dt * v
                
            elif sampler_type == "heun":
                k1 = get_velocity(x, t)
                k2 = get_velocity(x + dt * k1, t + dt)
                x = x + dt * 0.5 * (k1 + k2)
                
            elif sampler_type == "rk4":
                # 4th-order Runge-Kutta
                k1 = get_velocity(x, t)
                k2 = get_velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = get_velocity(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = get_velocity(x + dt * k3, t + dt)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
            elif sampler_type == "midpoint":
                # RK2 Midpoint method
                k1 = get_velocity(x, t)
                k2 = get_velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
                x = x + dt * k2

            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        x = x * self.inv_coord_scale
        
        # De-center using CA centroid (always (B, 3))
        if centers is not None:
            if self.decoder_coord_mode == "FULL":
                # Broadcast CA centroid to all 4 backbone atoms
                centers_expanded = centers.unsqueeze(1).unsqueeze(2).expand(
                    -1, -1, self.backbone_atom_count, -1
                )
                centers_flat = centers_expanded.reshape(batch_size, 1, self.decoder_coord_dim)
                x = x + centers_flat.to(x.device)
            else:
                x = x + centers.unsqueeze(1).to(x.device)
                
        if self.decoder_coord_mode == "FULL":
            return self._unflatten_backbone(x)
        return x


# Yeti is here. Please be nice.
YetiModel = StructureTokenizerFlowModel

