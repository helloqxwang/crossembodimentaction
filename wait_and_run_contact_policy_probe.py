from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_PID = 1700512
DEFAULT_INTERVAL_S = 10.0


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _build_command(extra_args: list[str]) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        "repr",
        "python",
        "train_contact_policy_probe.py",
        *extra_args,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for a PID to finish, then launch train_contact_policy_probe.py in conda env 'repr'."
    )
    parser.add_argument("--pid", type=int, default=DEFAULT_PID, help=f"PID to monitor. Default: {DEFAULT_PID}")
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help=f"Polling interval in seconds. Default: {DEFAULT_INTERVAL_S}",
    )
    parser.add_argument(
        "probe_args",
        nargs=argparse.REMAINDER,
        help="Optional extra args passed through to train_contact_policy_probe.py.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    probe_args = list(args.probe_args)
    if probe_args[:1] == ["--"]:
        probe_args = probe_args[1:]

    print(f"[{_timestamp()}] Monitoring pid={args.pid} every {args.interval:.1f}s")
    try:
        while _is_pid_running(int(args.pid)):
            print(f"[{_timestamp()}] pid={args.pid} is still running")
            time.sleep(max(0.1, float(args.interval)))
    except KeyboardInterrupt:
        print(f"[{_timestamp()}] Interrupted while waiting")
        return 130

    command = _build_command(probe_args)
    print(f"[{_timestamp()}] pid={args.pid} finished")
    print(f"[{_timestamp()}] Launching: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(project_dir))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
