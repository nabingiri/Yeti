# YETI Protein Structure Tokenizer

YETI learns discrete tokens from protein structures using a Transformer encoder, LFQ quantization, and a Hyena-based decoder.

## Quick Start

Install core dependencies:

```bash
conda env create -f yeti.yml

```

## Data Inputs

Set paths in `train_yeti.yaml`:

- `data.meta_data`: CSV/TSV with protein IDs
- `data.cif_dir`: directory with `{protein_id}.cif`
- `data.seq_dir`: directory with `{protein_id}.fasta` # not required currently.

## Train

```bash
python3 train_model.py --config-name train_yeti.yaml
```

Outputs are written to:

- `train.checkpoint_dir` (checkpoints)
- `train.log_dir` (logs)
- `wandb.logs` (optional W&B logs)

## Inference

```bash
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
```

Main outputs:

- token files (`tokens/`)
- embeddings (`embeddings_pre_quant/`, `embeddings_post_quant/`)
- reconstructed structures (`recon_coords/`)
- metrics (`rmsd_metrics.csv`, `entropy_perplexity_metrics.csv`)



More to come ... 

