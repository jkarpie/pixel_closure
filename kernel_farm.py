"""Bounded process farm for independent full-scale closure dataset kernels.

The inner PIXEL kernel pool helps only when one kernel has enough independent
blocks.  Most closure kernels do not, so assemble different datasets in spawned
processes instead.  Each dataset has its own cache path; a restarted run can
reuse kernels completed by earlier workers.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context


def _available_bytes() -> int | None:
    """Use the tighter of host and cgroup headroom on Linux."""
    available = None
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            with open(limit_path, encoding="ascii") as stream:
                limit = stream.read().strip()
            with open(used_path, encoding="ascii") as stream:
                used = int(stream.read().strip())
            if limit != "max":
                headroom = max(0, int(limit) - used)
                available = headroom if available is None else min(available, headroom)
        except (OSError, ValueError):
            pass
    return available


def worker_count(n_tasks: int) -> int:
    requested = os.environ.get("PIXEL_DATASET_WORKERS", "1").lower()
    if requested == "auto":
        requested = str(os.cpu_count() or 1)
    try:
        count = int(requested)
    except ValueError as exc:
        raise ValueError("PIXEL_DATASET_WORKERS must be a positive integer or auto") from exc
    if count < 1:
        raise ValueError("PIXEL_DATASET_WORKERS must be positive")
    count = min(count, n_tasks)
    available = _available_bytes()
    if available is not None:
        # Conservative default for a full 146-node dataset worker.  Both knobs
        # are adjustable for a machine with measured per-worker memory usage.
        gib = 1024 ** 3
        per_worker = float(os.environ.get("PIXEL_DATASET_WORKER_GIB", "4")) * gib
        reserve = float(os.environ.get("PIXEL_DATASET_RESERVE_GIB", "8")) * gib
        if per_worker <= 0 or reserve < 0:
            raise ValueError("dataset worker memory budget must be positive")
        count = min(count, max(1, int(max(0, available - reserve) // per_worker)))
    else:
        # No reliable available-memory source; don't launch an unbounded farm.
        count = min(count, 4)
    return count


def _init_worker() -> None:
    # The outer farm owns CPU parallelism; nested PIXEL pools would multiply
    # worker counts and memory use.  Spawn prevents inheriting JAX state.
    os.environ["PIXEL_KERNEL_WORKERS"] = "1"
    os.environ["PIXEL_ASSEMBLY_WORKERS"] = "1"


def _assemble(index, ds, fields, assembler):
    return index, ds.name, assembler(ds, fields)


def assemble_datasets(datasets, fields, assembler):
    """Return ordered ``(dataset name, operator)`` pairs, caching per dataset."""
    count = worker_count(len(datasets)) if datasets else 0
    if count <= 1:
        return [(ds.name, assembler(ds, fields)) for ds in datasets]
    print(f"kernel farm: {len(datasets)} datasets, {count} processes (inner workers=1)",
          flush=True)
    results = [None] * len(datasets)
    with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn"),
                             initializer=_init_worker) as pool:
        iterator = iter(enumerate(datasets))
        pending = {}

        def submit_next():
            try:
                index, ds = next(iterator)
            except StopIteration:
                return False
            pending[pool.submit(_assemble, index, ds, fields, assembler)] = index
            return True

        for _ in range(count):
            submit_next()
        completed = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                index, name, operator = future.result()
                results[index] = (name, operator)
                completed += 1
                print(f"kernel farm: {completed}/{len(datasets)} {name}", flush=True)
                submit_next()
    return results
