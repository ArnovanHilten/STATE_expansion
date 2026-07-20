import logging
import math

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer

from ..models import PerturbationModel
from .batch_speed_monitor import BatchSpeedMonitorCallback
from .model_flops_utilization import ModelFLOPSUtilizationCallback
from .cumulative_flops import CumulativeFLOPSCallback

logger = logging.getLogger(__name__)

__all__ = [
    "PerturbationModel",
    "BatchSpeedMonitorCallback",
    "ModelFLOPSUtilizationCallback",
    "CumulativeFLOPSCallback",
    "NaNLossCallback",
]


class NaNLossCallback(Callback):
    """Stops training immediately when a NaN or Inf loss is detected."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return
        val = loss.item() if hasattr(loss, "item") else float(loss)
        if math.isnan(val) or math.isinf(val):
            logger.error(f"NaN/Inf loss at batch {batch_idx} (loss={val}). Stopping training.")
            trainer.should_stop = True


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        pl_module.log("train/gradient_norm", gradient_norm(pl_module))


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm
