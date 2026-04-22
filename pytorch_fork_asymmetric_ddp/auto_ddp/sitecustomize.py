import os

# Auto-enable only when launcher sets TORCH_DDP_AUTO_WRAP=1.
if os.environ.get("TORCH_DDP_AUTO_WRAP", "0") == "1":
    from torch.distributed.asymmetric_autowrap import enable_from_env

    enable_from_env()

