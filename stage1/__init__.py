"""Stage-1 remote training: RPC worker closes forward/loss/backward/step; driver is CPU-only orchestration."""

from .driver_api import Stage1Trainer

__all__ = ["Stage1Trainer"]
