from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from ptycho.config import PipelineConfig, StageConfig
from ptycho.gpu_scheduler import GpuPool, GpuScheduler, Job


def build_single_slice_cfg(base: PipelineConfig, rotation_deg: float, iters: int, out_dir: Path) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    cfg.output.dir = str(out_dir)
    cfg.model.num_slices = 1
    cfg.model.slice_thickness = float(cfg.model.slice_thickness)
    cfg.preprocess.force_com_rotation = float(rotation_deg)
    cfg.stages = [
        StageConfig(
            name="single_slice",
            kind="pixelated",
            num_iters=int(iters),
            reset=True,
            batch_size=64,
            preprocess_batch_size=64,
            optimizer=base.stage("pixelated_baseline").optimizer,
            constraints=base.stage("pixelated_baseline").constraints,
        )
    ]
    return cfg


def main() -> int:
    base_run = Path("runs/pyrex_atomic_phase_test")
    diag_root = base_run / "diagnostics" / "single_slice_jobs"
    diag_root.mkdir(parents=True, exist_ok=True)

    base = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    base.validate()

    seed = 0
    gpus = [5, 6, 7]

    jobs: list[Job] = []

    baseline_dir = diag_root / "baseline_rot_p73_iter50"
    baseline_cfg = build_single_slice_cfg(base, rotation_deg=73.0, iters=50, out_dir=baseline_dir)
    baseline_cfg_path = baseline_dir / "config.yaml"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_cfg.to_yaml(baseline_cfg_path)
    jobs.append(
        Job(
            name="single_slice_baseline_rot_p73_iter50",
            argv=[
                sys.executable,
                "-m",
                "ptycho.worker",
                "-c",
                str(baseline_cfg_path),
                "--stage",
                "single_slice",
                "-o",
                str(baseline_dir),
                "--gpu-label",
                "baseline+73",
            ],
            deps=[],
            log_path=baseline_dir / "worker.log",
        )
    )

    candidates = [73.0, -73.0, 17.0, -17.0, 163.0, -163.0]
    for rot in candidates:
        tag = f"p{int(rot)}" if rot >= 0 else f"m{abs(int(rot))}"
        out_dir = diag_root / f"rot_{tag}_iter15"
        cfg = build_single_slice_cfg(base, rotation_deg=rot, iters=15, out_dir=out_dir)
        cfg_path = out_dir / "config.yaml"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(cfg_path)
        jobs.append(
            Job(
                name=f"rot_{tag}_iter15",
                argv=[
                    sys.executable,
                    "-m",
                    "ptycho.worker",
                    "-c",
                    str(cfg_path),
                    "--stage",
                    "single_slice",
                    "-o",
                    str(out_dir),
                    "--gpu-label",
                    f"rot{rot:+g}",
                ],
                deps=[],
                log_path=out_dir / "worker.log",
            )
        )

    pool = GpuPool(allowed=gpus, min_free_mb=8000)

    def env_builder(gpu: int) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "4",
            "PYREX_SEED": str(seed),
        }

    scheduler = GpuScheduler(pool, log_path=diag_root / "gpu_assignment.log", env_builder=env_builder)
    result = scheduler.run(jobs)

    payload = {
        "ok": bool(result.ok),
        "aborted": result.aborted,
        "jobs": [
            {
                "name": j.name,
                "gpu": j.gpu,
                "returncode": j.returncode,
                "runtime_s": j.runtime,
                "log": str(j.log_path) if j.log_path else None,
            }
            for j in jobs
        ],
    }
    (diag_root / "job_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
