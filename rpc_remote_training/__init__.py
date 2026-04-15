"""Baseline remote RPC training: worker runs forward/loss/backward/optimizer.step; driver is CPU-side orchestration."""

from .driver_api import Stage1Trainer

__all__ = ["Stage1Trainer"]
