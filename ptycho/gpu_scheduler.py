"""Coarse-grained GPU scheduler for the multislice ptychography stages.

``quantem``'s diffractive-imaging backend has **no** multi-GPU / DDP support
(its DDP helper is tomography-only), so a single reconstruction always lives on
one card. Rather than rewriting the numerical core, this scheduler parallelises
*independent stages of the reconstruction DAG*: every stage runs as its own
subprocess pinned to one physical GPU through ``CUDA_VISIBLE_DEVICES``.

Dependencies come straight from ``StageConfig.init_from``: a stage may only
start once its checkpoint exists, so continuation semantics are never broken.

Selection policy: among the allowed GPUs, require enough free memory and pick
the currently least-loaded one. Assignments and timings are appended to
``gpu_assignment.log`` and collected into ``stage_metrics.json``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

# Escape hatches for wedged drivers (nvidia-smi stuck in uninterruptible sleep).
#   PTYCHO_GPU_QUERY=off   -> never spawn nvidia-smi, use the static fallback
#   PTYCHO_GPU_STATIC=5,6,7-> GPUs assumed idle (defaults to the pool's allowed set)
#   PTYCHO_GPU_TIMEOUT=5   -> per-query timeout in seconds
_GPU_TIMEOUT = float(os.environ.get("PTYCHO_GPU_TIMEOUT", "5"))
_STATIC_GPUS_ENV = "PTYCHO_GPU_STATIC"


def _static_gpus_from_env() -> list[int] | None:
    """Parse ``PTYCHO_GPU_STATIC`` (e.g. ``5,6,7``); ``None`` when unset/blank."""
    raw = os.environ.get(_STATIC_GPUS_ENV, "").strip()
    if not raw:
        return None
    return [int(part) for part in raw.replace(";", ",").split(",") if part.strip()]


@dataclass
class GpuState:
    index: int
    memory_used_mb: int
    memory_total_mb: int
    utilization: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


class GpuPool:
    """Allowed-GPU pool with free-memory checks and least-loaded selection."""

    def __init__(
        self,
        allowed: Iterable[int],
        min_free_mb: int = 8000,
        query: Callable[[], dict[int, GpuState]] | None = None,
    ) -> None:
        self.allowed = [int(g) for g in allowed]
        self.min_free_mb = int(min_free_mb)
        if query is not None:
            self._query = query
        else:
            static = _static_gpus_from_env()
            if static is not None:
                logger.warning(
                    "GPU query bypassed via %s=%s -> assuming GPUs %s are idle",
                    _STATIC_GPUS_ENV,
                    os.environ.get(_STATIC_GPUS_ENV),
                    static,
                )
                self._query = lambda: _static_gpu_states(static)
            elif os.environ.get("PTYCHO_GPU_QUERY", "").lower() in {
                "off",
                "0",
                "false",
                "no",
            }:
                logger.warning(
                    "GPU query disabled via PTYCHO_GPU_QUERY -> assuming GPUs %s are idle",
                    self.allowed,
                )
                self._query = lambda: _static_gpu_states(self.allowed)
            else:
                self._query = _query_nvidia_smi
        self._reserved: dict[int, str] = {}

    def states(self) -> dict[int, GpuState]:
        return self._query()

    def available(self) -> list[tuple[GpuState, int]]:
        """Return (state, live_job_count) for GPUs that can take another job."""

        states = self.states()
        out: list[tuple[GpuState, int]] = []
        for gpu in self.allowed:
            st = states.get(gpu)
            if st is None:
                logger.warning("GPU %s not reported by nvidia-smi", gpu)
                continue
            live = self._count_live(gpu)
            if live >= 1:
                continue
            if st.memory_free_mb < self.min_free_mb:
                logger.info(
                    "GPU %s skipped: only %d MiB free (< %d MiB)",
                    gpu,
                    st.memory_free_mb,
                    self.min_free_mb,
                )
                continue
            out.append((st, live))
        # Least loaded first: running jobs, then utilisation, then free memory.
        out.sort(key=lambda item: (item[1], item[0].utilization, -item[0].memory_free_mb))
        return [st for st, _ in out]

    def reserve(self, gpu: int, job: str) -> None:
        self._reserved[gpu] = job

    def release(self, gpu: int) -> None:
        self._reserved.pop(gpu, None)

    def _count_live(self, gpu: int) -> int:
        return 1 if gpu in self._reserved else 0


def _query_nvidia_smi(timeout: float | None = None) -> dict[int, GpuState]:
    """Read per-GPU memory / utilisation from ``nvidia-smi``.

    NOTE: ``subprocess.run(..., timeout=...)`` must **not** be used here. On
    timeout it calls ``kill()`` and then an unbounded ``wait()``; when the
    driver is wedged ``nvidia-smi`` sits in uninterruptible sleep (state ``D``),
    SIGKILL cannot be delivered and ``wait()`` blocks forever - the 5 s timeout
    never fires and the whole scheduler hangs. We therefore drive ``Popen``
    directly and simply *abandon* the child on timeout instead of reaping it.
    """
    timeout = _GPU_TIMEOUT if timeout is None else timeout
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "nvidia-smi not found. GPU scheduler requires NVIDIA driver tools."
        ) from exc

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Best effort only: a D-state child ignores SIGKILL until the driver
        # recovers, and waiting for it would hang us forever.
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 - child may already be gone
            pass
        raise RuntimeError(
            f"nvidia-smi timed out after {timeout}s (driver/hardware query hang). "
            f"Set {_STATIC_GPUS_ENV}=5,6,7 (or PTYCHO_GPU_QUERY=off) to bypass "
            "the query. Stuck nvidia-smi processes may need a node reboot."
        ) from None

    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {(stderr or '').strip()}")

    states: dict[int, GpuState] = {}
    for line in (stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, used, total, util = parts[:4]
        states[int(idx)] = GpuState(int(idx), int(used), int(total), int(util))
    return states


def _static_gpu_states(gpus: Iterable[int]) -> dict[int, GpuState]:
    """Fabricate 'idle' states so scheduling can proceed without querying the driver."""
    return {
        int(gpu): GpuState(int(gpu), memory_used_mb=0, memory_total_mb=1 << 30, utilization=0)
        for gpu in gpus
    }


@dataclass(eq=False)
class Job:
    """One schedulable unit: a reconstruction stage running in a subprocess."""

    name: str
    argv: list[str]
    deps: list[str] = field(default_factory=list)
    gpu: int | None = None
    start: float | None = None
    end: float | None = None
    returncode: int | None = None
    log_path: Path | None = None

    @property
    def runtime(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return self.end - self.start


class JobResult:
    def __init__(self, jobs: list[Job], aborted: str | None) -> None:
        self.jobs = jobs
        self.aborted = aborted

    @property
    def ok(self) -> bool:
        return self.aborted is None

    def by_name(self) -> dict[str, Job]:
        return {j.name: j for j in self.jobs}


class GpuScheduler:
    """Run independent jobs concurrently on the allowed GPUs."""

    def __init__(
        self,
        pool: GpuPool,
        log_path: Path,
        poll_interval: float = 2.0,
        env_builder: Callable[[int], dict[str, str]] | None = None,
    ) -> None:
        self.pool = pool
        self.log_path = Path(log_path)
        self.poll_interval = poll_interval
        self._env_builder = env_builder or _cuda_env
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._running: dict[Job, subprocess.Popen] = {}

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        logger.info(message)

    def run(self, jobs: list[Job]) -> JobResult:
        """Execute ``jobs`` respecting ``deps``; abort on first failure."""
        self.log_path.write_text("", encoding="utf-8")
        self._log(
            f"scheduler start: allowed GPUs={self.pool.allowed} "
            f"min_free_mb={self.pool.min_free_mb} jobs={[j.name for j in jobs]}"
        )

        pending = {j.name: j for j in jobs}
        done: set[str] = set()
        failed: set[str] = set()

        while pending or self._running:
            for job in list(self._running.keys()):
                proc = self._running[job]
                if proc.poll() is None:
                    continue
                job.returncode = proc.returncode
                job.end = time.time()
                del self._running[job]
                if job.gpu is not None:
                    self.pool.release(job.gpu)
                if job.returncode == 0:
                    done.add(job.name)
                    self._log(
                        f"DONE    {job.name} gpu={job.gpu} runtime={job.runtime:.1f}s"
                    )
                else:
                    failed.add(job.name)
                    self._log(
                        f"FAILED  {job.name} gpu={job.gpu} rc={job.returncode} "
                        f"runtime={job.runtime:.1f}s log={job.log_path}"
                    )
                    return self._finish(jobs, failed)

            blocked = self._blocked_names(pending, done, failed)
            for job in list(pending.values()):
                if job.name in self._running or job.name in blocked:
                    continue
                try:
                    candidates = self.pool.available()
                except Exception as exc:
                    self._log(f"ABORT   GPU query failed: {exc}")
                    return self._finish(jobs, failed | set(pending.keys()))
                if not candidates:
                    break
                gpu = candidates[0]
                job.gpu = gpu.index
                job.start = time.time()
                self.pool.reserve(gpu.index, job.name)
                proc = self._spawn(job, gpu.index)
                self._running[job] = proc
                pending.pop(job.name)
                self._log(
                    f"START   {job.name} gpu={gpu.index} "
                    f"free={gpu.memory_free_mb} MiB util={gpu.utilization}% deps={job.deps}"
                )

            if not self._running:
                if pending:
                    blocked_now = self._blocked_names(pending, done, failed)
                    stuck = [n for n in pending if n in blocked_now]
                    if stuck:
                        self._log(f"ABORT   dependencies failed for {stuck}")
                        return self._finish(jobs, set(stuck) | failed)
                    self._log("IDLE    no GPU available and no running job")
                    time.sleep(self.poll_interval)
                continue
            time.sleep(self.poll_interval)

        return self._finish(jobs, failed)

    @staticmethod
    def _blocked_names(
        pending: dict[str, Job], done: set[str], failed: set[str]
    ) -> set[str]:
        """Names whose deps are not satisfied *yet* (vs. permanently broken)."""
        blocked = set()
        for name, job in pending.items():
            if any(dep in failed for dep in job.deps):
                blocked.add(name)
            elif not all(dep in done for dep in job.deps):
                blocked.add(name)
        return blocked

    def _spawn(self, job: Job, gpu: int) -> subprocess.Popen:
        import os

        env = os.environ.copy()
        env.update(self._env_builder(gpu))
        job.log_path = Path(f"{job.log_path}")
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(job.log_path, "w", encoding="utf-8")
        return subprocess.Popen(
            job.argv, stdout=handle, stderr=subprocess.STDOUT, env=env
        )

    def _finish(self, jobs: list[Job], failed: set[str]) -> JobResult:
        for job, proc in list(self._running.items()):
            proc.terminate()
            job.returncode = proc.wait()
            job.end = time.time()
            self._log(f"KILLED  {job.name} (dependency failure)")
        self._running.clear()
        aborted = ",".join(sorted(failed)) if failed else None
        self._log(f"scheduler stop: done_ok aborted={aborted}")
        return JobResult(jobs, aborted)


def _cuda_env(gpu: int) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "4",
    }


def write_stage_metrics(
    path: Path, jobs: list[Job], per_stage_metrics: dict[str, dict[str, Any]]
) -> None:
    """Merge scheduler timings with the worker-reported metrics and save JSON."""
    payload: dict[str, Any] = {}
    for job in jobs:
        entry = dict(per_stage_metrics.get(job.name, {}))
        entry.update(
            {
                "gpu": job.gpu,
                "start": job.start,
                "end": job.end,
                "runtime_s": round(job.runtime, 2),
                "returncode": job.returncode,
                "log": str(job.log_path) if job.log_path else None,
                "deps": job.deps,
            }
        )
        payload[job.name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
