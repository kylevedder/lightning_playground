import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models
import losses
import val_metrics
from pathlib import Path

import pytorch_lightning as pl
from typing import Dict, List, Tuple, Any
import nntime
import time
from config_wrapper import ConfigWrapper


class ModelWrapper(pl.LightningModule):

    def __init__(self, cfg: ConfigWrapper):
        super().__init__()
        self.cfg = cfg
        self.model = getattr(models, cfg.model.name)(**cfg.model.args)

        self.metric = getattr(val_metrics, cfg.metric.name)(**cfg.metric.args)

        if self.cfg.get_or_default("is_trainable", True):
            self.loss_fn = getattr(losses,
                                   cfg.loss_fn.name)(**cfg.loss_fn.args)
        else:
            # log that we are not using a loss function
            print("RUNNING AS INFERENCE ONLY. NOT USING A LOSS FUNCTION.")

        self.lr = cfg.learning_rate

        self.train_forward_args = self.cfg.get_or_default(
            "train_forward_args", {})
        self.val_forward_args = self.cfg.get_or_default("val_forward_args", {})
        self.has_labels = self.cfg.get_or_default("has_labels", True)

        self.save_output_folder = self.cfg.get_or_default(
            "save_output_folder", None)

    def on_load_checkpoint(self, checkpoint):
        checkpoint_lrs = set()

        for optimizer_state_idx in range(len(checkpoint['optimizer_states'])):
            for param_group_idx in range(
                    len(checkpoint['optimizer_states'][optimizer_state_idx]
                        ['param_groups'])):
                checkpoint_lrs.add(
                    checkpoint['optimizer_states'][optimizer_state_idx]
                    ['param_groups'][param_group_idx]['lr'])

        # If there are multiple learning rates, or if the learning rate is not the same as the one in the config, reset the optimizer.
        # This is to handle the case where we want to resume training with a different learning rate.

        reset_learning_rate = (len(set(checkpoint_lrs)) !=
                               1) or (self.lr != list(checkpoint_lrs)[0])

        if reset_learning_rate:
            print("Resetting learning rate to the one in the config.")
            checkpoint.pop('optimizer_states')
            checkpoint.pop('lr_schedulers')

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer

    def training_step(self, input_batch: Dict[str, Any], batch_idx):
        model_res = self.model(input_batch, **self.train_forward_args)
        loss_res = self.loss_fn(input_batch, model_res)
        assert isinstance(
            loss_res,
            dict), f"loss_res should be a dict. Got {type(loss_res)}."
        assert "loss" in loss_res, f"loss not in loss_res keys {loss_res.keys()}"
        loss = loss_res.pop("loss")
        self.log("train/loss", loss, on_step=True)
        for k, v in loss_res.items():
            self.log(f"train/{k}", v, on_step=True)
        return {"loss": loss}

    def validation_step(self, input_batch: Dict[str, Any], batch_idx):
        nntime.timer_start(self, "validation_forward")
        start_time = time.time()
        model_res = self.model(input_batch, **self.val_forward_args)
        end_time = time.time()
        nntime.timer_end(self, "validation_forward")
        self.metric.to(self.device)

        if self.save_output_folder is not None:
            self._save_output(input_batch, model_res, batch_idx,
                              end_time - start_time)

        if not self.has_labels:
            return

        self.metric.update(input_batch, model_res, batch_idx,
                           end_time - start_time)

    def _save_output(self, input_batch: Dict[str, Any],
                     output_batch: Dict[str, Any], batch_idx: int,
                     forward_time: float):
        pass

    def _to_numpy(self, d):
        if isinstance(d, torch.Tensor):
            return d.cpu().numpy()
        if isinstance(d, list):
            return [self._to_numpy(x) for x in d]
        if isinstance(d, tuple):
            return tuple(self._to_numpy(x) for x in d)
        if isinstance(d, dict):
            return {k: self._to_numpy(v) for k, v in d.items()}
        return d

    def _save_validation_data(self, validation_result_dict: dict):
        pass

    def _log_validation_metrics(self, validation_result_dict: dict):
        assert self.global_rank == 0, "Only rank 0 should log validation metrics."
        assert isinstance(validation_result_dict,
                          dict), "validation_result_dict should be a dict."

        for k, v in validation_result_dict.items():
            self.log(f"val/{k}", v, sync_dist=False, rank_zero_only=True)

    def validation_epoch_end(self, batch_parts):
        import time
        before_gather = time.time()

        # These are copies of the metric values on each rank.
        validation_result_dict = self.metric.gather(self.all_gather)

        after_gather = time.time()

        print(
            f"Rank {self.global_rank} gathers done in {after_gather - before_gather}."
        )

        # Reset the metric for the next epoch. We have to do this on each rank, and because we are using
        # copies of the metric values above, we don't have to worry about over-writing the values.
        self.metric.reset()

        if self.global_rank != 0:
            return {}

        self._log_validation_metrics(validation_result_dict)

        validation_result_dict = self._to_numpy(validation_result_dict)

        self._save_validation_data(validation_result_dict)

        return {}