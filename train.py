import os
# PL_FAULT_TOLERANT_TRAINING=1
# to enable fault tolerant training
#os.environ['PL_FAULT_TOLERANT_TRAINING'] = '1'

import datetime
import torch
from pathlib import Path
import argparse

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import dataloaders
from model_wrapper import ModelWrapper
from config_wrapper import ConfigWrapper


def get_rank() -> int:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_checkpoint_path(cfg, checkpoint_dir_name: str):
    cfg_filename = Path(cfg.filename)
    config_name = cfg_filename.stem
    parent_name = cfg_filename.parent.name
    parent_path = Path(f"model_checkpoints/{parent_name}/{config_name}/")
    rank = get_rank()
    if rank == 0:
        # Since we're rank 0, we can create the directory
        return parent_path / checkpoint_dir_name, checkpoint_dir_name
    else:
        # Since we're not rank 0, we shoulds grab the most recent directory
        checkpoint_path = sorted(parent_path.glob("*"))[-1]
        return checkpoint_path, checkpoint_path.name


def make_dataloader(cfg, dataset_cfg_key: str, dataloader_cfg_key: str):
    dataset = cfg[dataset_cfg_key]
    dataloader = cfg[dataloader_cfg_key]
    # Handle single loader case
    if not isinstance(dataset, list):
        train_dataset = getattr(dataloaders, dataset.name)(**dataset.args)
        return torch.utils.data.DataLoader(train_dataset, **dataloader.args)

    # Handle multiple loader case
    assert isinstance(dataset, list), \
        "Single dataset not specified, but the config file does not specify a list of datasets."

    print("Using multiple datasets of length:", len(dataset))
    train_dataloader_lst = [
        getattr(dataloaders, d.name)(**d.args) for d in dataset
    ]

    # Use the concat dataloader to combine the multiple dataloaders
    concat_dataset = torch.utils.data.ConcatDataset(train_dataloader_lst)
    return torch.utils.data.DataLoader(concat_dataset, **dataloader.args)


def make_train_dataloader(cfg):
    return make_dataloader(cfg, "dataset", "dataloader")


def make_val_dataloader(cfg):
    return make_dataloader(cfg, "test_dataset", "test_dataloader")

def main():

    # Get config file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--resume_from_checkpoint', type=Path, default=None)
    parser.add_argument(
        '--checkpoint_dir_name',
        type=str,
        default=datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist"
    cfg = ConfigWrapper.fromfile(args.config)

    if hasattr(cfg, "is_trainable") and not cfg.is_trainable:
        raise ValueError("Config file indicates this model is not trainable.")

    if hasattr(cfg, "seed_everything"):
        pl.seed_everything(cfg.seed_everything)

    resume_from_checkpoint = args.resume_from_checkpoint

    checkpoint_path, checkpoint_dir_name = get_checkpoint_path(
        cfg, args.checkpoint_dir_name)
    if not args.dry_run:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        # Save config file to checkpoint directory
        cfg.dump(str(checkpoint_path / "config.py"))

    tbl = TensorBoardLogger("tb_logs",
                            name=cfg.filename,
                            version=checkpoint_dir_name)

    train_dataloader = make_train_dataloader(cfg)
    val_dataloader = make_val_dataloader(cfg)

    print("Train dataloader length:", len(train_dataloader))
    print("Val dataloader length:", len(val_dataloader))

    model = ModelWrapper(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="checkpoint_{epoch:03d}_{step:010d}",
        save_top_k=-1,
        # every_n_train_steps=cfg.save_every,
        every_n_epochs=1,
        save_on_train_epoch_end=True)

    trainer = pl.Trainer(devices=args.gpus,
                         accelerator="gpu",
                         logger=tbl,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         move_metrics_to_cpu=False,
                         num_sanity_val_steps=2,
                         log_every_n_steps=2,
                         val_check_interval=cfg.validate_every,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch
                         if hasattr(cfg, "check_val_every_n_epoch") else 1,
                         max_epochs=cfg.epochs,
                         resume_from_checkpoint=resume_from_checkpoint,
                         accumulate_grad_batches=cfg.accumulate_grad_batches
                         if hasattr(cfg, "accumulate_grad_batches") else 1,
                         gradient_clip_val=cfg.gradient_clip_val if hasattr(
                             cfg, "gradient_clip_val") else 0.0,
                         callbacks=[checkpoint_callback])
    if args.dry_run:
        print("Dry run, exiting")
        exit(0)
    print("Starting training")
    print("Length of train dataloader:", len(train_dataloader))
    print("Length of val dataloader:", len(val_dataloader))
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()