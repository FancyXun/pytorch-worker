#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
import traceback


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: launch_user.py <user_train.py> [script_args...]", file=sys.stderr)
        return 2

    script_path = sys.argv[1]
    script_args = sys.argv[2:]

    try:
        from torch.distributed.asymmetric_autowrap import enable_from_env

        enable_from_env()
    except Exception:
        print("[auto-ddp] failed to initialize distributed runtime; aborting.", file=sys.stderr)
        traceback.print_exc()
        return 1

    sys.argv = [script_path, *script_args]
    runpy.run_path(script_path, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

