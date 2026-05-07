"""
__author__: Nabin
Train the Yeti model using the Hyena architecture.
"""

import os
import sys
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

torch.set_float32_matmul_precision("medium")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datamodule import YetiDataModule
from structure_tokenizer_model import YetiModel


def setup_strategy(cfg: DictConfig):
    """
    Setup the strategy for the training. This is used to distribute the training across multiple GPUs.
    """
    if cfg.train.strategy in ("ddp", "ddp_find_unused_parameters_false"):
        return DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True, static_graph=False)
    if cfg.train.strategy == "ddp_find_unused_parameters_true":
        return DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=False)
    return cfg.train.strategy


@hydra.main(config_path=".", config_name="yeti", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed, workers=True)
    print(f"Seed: {cfg.train.seed}")
    print(f"Config: {OmegaConf.to_yaml(cfg)}")

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    os.makedirs(cfg.wandb.logs, exist_ok=True)

    # Initialize the data module and run a small sanity check on the first sample
    data = YetiDataModule(cfg)
    data.setup("fit")

    model = YetiModel(cfg)
    print("Model : ", model)
    

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.train.checkpoint_dir,
            monitor=cfg.train.monitor,
            mode=cfg.train.mode,
            filename='yeti-{step:06d}-{val_loss:.6f}', # change this for valid or train loss.
            every_n_train_steps=500,  # Save every 500 steps for regular backups
            save_top_k=3,  # Keep last 3 checkpoints
            save_last=True,  # Plus always keep the most recent
        ),
        LearningRateMonitor(logging_interval='step', log_momentum=False),
    ]

    logger = None
    if getattr(cfg.wandb, 'project_name', None):
        logger = WandbLogger(
            project=cfg.wandb.project_name,
            entity=getattr(cfg.wandb, 'entity_name', None),
            save_dir=cfg.wandb.logs,
            log_model=True,
            name=getattr(cfg.wandb, 'run_name', None),
        )
        try:
            logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
        except Exception:
            pass

    # Setup the strategy for the training.
    strategy_cfg = setup_strategy(cfg)

    trainer = Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.num_gpu,
        num_nodes=cfg.train.num_nodes,
        precision=cfg.train.precision,
        max_epochs=cfg.train.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.train.log_interval,
        enable_checkpointing=True,
        strategy=strategy_cfg,
        use_distributed_sampler=False,
        enable_progress_bar=False,
        gradient_clip_val=getattr(cfg.train, 'gradient_clip_val', 0.0),
    )

    ckpt_path = cfg.train.resume_from_checkpoint or None
    trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)

    if trainer.is_global_zero:
        final_path = os.path.join(cfg.train.checkpoint_dir, 'final_flow_tokenizer.ckpt')
        trainer.save_checkpoint(final_path)
        print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()


