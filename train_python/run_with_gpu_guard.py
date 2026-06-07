#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


BYTES_PER_GB = 1024 ** 3
CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}
CACHE_FILE_SUFFIXES = {".pyc", ".pyo"}
SKIP_SCAN_DIR_NAMES = {".git", ".codegraph", "outputs", "build", "research_pack_2026-06-03"}
SKIP_SCAN_DIR_PREFIXES = (".venv",)


def resolve_nvidia_smi() -> str:
    env_path = os.environ.get("NVIDIA_SMI_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if sys.platform.startswith("win"):
        roots = [
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "DriverStore" / "FileRepository",
            Path(r"C:\Windows\System32\DriverStore\FileRepository"),
        ]
        candidates: list[Path] = []
        for root in roots:
            if root.exists():
                candidates.extend(root.glob("*/nvidia-smi.exe"))
        if candidates:
            newest = max(candidates, key=lambda item: item.stat().st_mtime)
            return str(newest)
    return "nvidia-smi"


def command_max_length(command: list[str]) -> int | None:
    for i, token in enumerate(command):
        if token == "--max-length" and i + 1 < len(command):
            return int(command[i + 1])
        if token.startswith("--max-length="):
            return int(token.split("=", 1)[1])
    return None


def timeout_expired(start_time: float, timeout_seconds: float, now: float | None = None) -> bool:
    if timeout_seconds <= 0.0:
        return False
    current = time.time() if now is None else now
    return current - start_time >= timeout_seconds


def start_memory_allowed(state: dict, max_start_memory_ratio: float) -> bool:
    if max_start_memory_ratio <= 0.0:
        return True
    return float(state["memory_used_ratio"]) <= max_start_memory_ratio


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def should_skip_cleanup_scan(dirname: str) -> bool:
    return dirname in SKIP_SCAN_DIR_NAMES or dirname.startswith(SKIP_SCAN_DIR_PREFIXES)


def disk_state(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "total_gb": usage.total / BYTES_PER_GB,
        "used_gb": usage.used / BYTES_PER_GB,
        "free_gb": usage.free / BYTES_PER_GB,
    }


def cleanup_repo_caches(root: Path, *, dry_run: bool = False) -> dict:
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"cleanup root does not exist or is not a directory: {root}")

    targets: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)
        kept_dirnames = []
        for dirname in dirnames:
            child = current / dirname
            if dirname in CACHE_DIR_NAMES:
                targets.append(child)
                continue
            if should_skip_cleanup_scan(dirname):
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

        for filename in filenames:
            child = current / filename
            if child.suffix in CACHE_FILE_SUFFIXES:
                targets.append(child)

    # Avoid deleting files inside a directory target twice.
    directory_targets = [target for target in targets if target.is_dir()]
    filtered: list[Path] = []
    for target in targets:
        if target.is_file() and any(parent in directory_targets for parent in target.parents):
            continue
        filtered.append(target)

    removed: list[dict] = []
    errors: list[dict] = []
    estimated_bytes = 0
    for target in sorted(filtered, key=lambda item: str(item)):
        try:
            size = path_size_bytes(target)
            estimated_bytes += size
            removed.append({"path": str(target), "bytes": size})
            if dry_run:
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except OSError as exc:
            errors.append({"path": str(target), "error": str(exc)})

    return {
        "root": str(root),
        "dry_run": dry_run,
        "target_count": len(filtered),
        "estimated_bytes": estimated_bytes,
        "estimated_mib": estimated_bytes / (1024 ** 2),
        "targets": removed,
        "errors": errors,
    }


def query_gpu() -> dict:
    proc = subprocess.run(
        [
            resolve_nvidia_smi(),
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")
    first = proc.stdout.strip().splitlines()[0]
    util, used, total = [int(part.strip()) for part in first.split(",")[:3]]
    return {
        "utilization_gpu_pct": util,
        "memory_used_mib": used,
        "memory_total_mib": total,
        "memory_used_ratio": used / max(total, 1),
    }


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform.startswith("win"):
        proc.terminate()
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command while enforcing a GPU memory guard.")
    parser.add_argument("--max-memory-ratio", type=float, default=0.85)
    parser.add_argument(
        "--max-start-memory-ratio",
        type=float,
        default=0.0,
        help="Reject before launch if current GPU memory ratio is above this value; use 0 to disable.",
    )
    parser.add_argument(
        "--min-disk-free-gb",
        type=float,
        default=0.0,
        help="Reject the command before launch if the selected disk has less free space than this.",
    )
    parser.add_argument(
        "--disk-check-path",
        type=Path,
        default=Path("."),
        help="Path whose containing filesystem is measured before and after the guarded command.",
    )
    parser.add_argument(
        "--cleanup-repo-caches",
        action="store_true",
        help="After the command exits, remove only repo-local Python/test caches under --cleanup-root.",
    )
    parser.add_argument("--cleanup-root", type=Path, default=Path("."))
    parser.add_argument("--cleanup-dry-run", action="store_true")
    parser.add_argument(
        "--max-length-ceiling",
        type=int,
        default=96,
        help="Reject guarded eval commands whose --max-length exceeds this ceiling; use 0 to disable.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=0.0,
        help="Terminate the guarded command after this many seconds; use 0 to disable.",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("missing command after --")
    if not 0.0 < args.max_memory_ratio <= 1.0:
        raise SystemExit("--max-memory-ratio must be in (0, 1]")
    if not 0.0 <= args.max_start_memory_ratio <= 1.0:
        raise SystemExit("--max-start-memory-ratio must be in [0, 1]")
    if args.max_length_ceiling < 0:
        raise SystemExit("--max-length-ceiling must be >= 0")
    if args.min_disk_free_gb < 0:
        raise SystemExit("--min-disk-free-gb must be >= 0")
    if args.timeout_sec < 0:
        raise SystemExit("--timeout-sec must be >= 0")
    requested_max_length = command_max_length(args.command)
    if args.max_length_ceiling and requested_max_length and requested_max_length > args.max_length_ceiling:
        raise SystemExit(
            f"guard rejected --max-length {requested_max_length}; "
            f"ceiling is {args.max_length_ceiling}"
        )

    disk_path = args.disk_check_path.resolve()
    start_disk_state = disk_state(disk_path)
    if args.min_disk_free_gb and start_disk_state["free_gb"] < args.min_disk_free_gb:
        raise SystemExit(
            f"disk free space below guard: {start_disk_state['free_gb']:.2f} GB "
            f"< {args.min_disk_free_gb:.2f} GB at {disk_path}"
        )

    samples: list[dict] = []
    start_state = query_gpu()
    if not start_memory_allowed(start_state, args.max_start_memory_ratio):
        raise SystemExit(
            f"GPU memory above start guard: {start_state['memory_used_mib']}/"
            f"{start_state['memory_total_mib']} MiB "
            f"({start_state['memory_used_ratio']:.4f} > {args.max_start_memory_ratio:.4f})"
        )
    if start_state["memory_used_ratio"] > args.max_memory_ratio:
        raise SystemExit(
            f"GPU memory already above guard: {start_state['memory_used_mib']}/"
            f"{start_state['memory_total_mib']} MiB"
        )

    proc = subprocess.Popen(args.command)
    killed = False
    killed_by_timeout = False
    command_start_time = time.time()
    try:
        while proc.poll() is None:
            sample = query_gpu()
            sample["t_seconds"] = time.time()
            samples.append(sample)
            if sample["memory_used_ratio"] > args.max_memory_ratio:
                killed = True
                terminate(proc)
                break
            if timeout_expired(command_start_time, args.timeout_sec, sample["t_seconds"]):
                killed_by_timeout = True
                terminate(proc)
                break
            time.sleep(args.poll_seconds)
    finally:
        terminate(proc)

    end_state = query_gpu()
    cleanup_result = None
    if args.cleanup_repo_caches:
        cleanup_result = cleanup_repo_caches(args.cleanup_root, dry_run=args.cleanup_dry_run)
    end_disk_state = disk_state(disk_path)
    max_sample = max(samples, key=lambda item: item["memory_used_ratio"], default=start_state)
    result = {
        "command": args.command,
        "returncode": proc.returncode,
        "killed_by_guard": killed,
        "killed_by_timeout": killed_by_timeout,
        "timeout_seconds": args.timeout_sec,
        "max_memory_used_mib": max_sample["memory_used_mib"],
        "memory_total_mib": max_sample["memory_total_mib"],
        "max_memory_used_ratio": max_sample["memory_used_ratio"],
        "max_utilization_gpu_pct": max((item["utilization_gpu_pct"] for item in samples), default=start_state["utilization_gpu_pct"]),
        "start_state": start_state,
        "end_state": end_state,
        "start_disk_state": start_disk_state,
        "end_disk_state": end_disk_state,
        "post_cleanup": cleanup_result,
        "samples": samples,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if killed:
        raise SystemExit(90)
    if killed_by_timeout:
        raise SystemExit(91)
    raise SystemExit(proc.returncode or 0)


if __name__ == "__main__":
    main()
