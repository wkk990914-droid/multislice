from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

from ptycho.config import PipelineConfig, StageConfig
from ptycho.gpu_scheduler import GpuPool, GpuScheduler, Job


def build_single_slice_cfg(
    base: PipelineConfig,
    rotation_deg: float,
    iters: int,
    out_dir: Path,
    dp_crop: tuple[int, int, int, int],
) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    cfg.output.dir = str(out_dir)
    cfg.model.num_slices = 1
    cfg.preprocess.force_com_rotation = float(rotation_deg)
    cfg.data.dp_crop = tuple(int(v) for v in dp_crop)
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


def crop_for_center(center: tuple[float, float], size: int = 192) -> tuple[int, int, int, int]:
    cx, cy = center
    half = size / 2.0
    qx0 = int(np.round(cx - half))
    qy0 = int(np.round(cy - half))
    return (qy0, qy0 + size, qx0, qx0 + size)


def main() -> int:
    base_run = Path("runs/pyrex_atomic_phase_test")
    diag_root = base_run / "diagnostics" / "center_sweep" / "fine"
    diag_root.mkdir(parents=True, exist_ok=True)

    base = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    base.validate()

    summary_path = base_run / "diagnostics" / "center_sweep_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}; run coarse sweep analysis first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best")
    if not best:
        raise SystemExit("no best center in summary")

    best_dx = int(best["dx"])
    best_dy = int(best["dy"])
    center_com = summary["center_com"]
    center_base = (float(center_com["cx"]) + best_dx, float(center_com["cy"]) + best_dy)

    theta0 = 17.0
    iters = 18
    seed = 0
    gpus = [5, 6, 7]

    grid = [-1, 0, 1]
    jobs: list[Job] = []
    for ddy in grid:
        for ddx in grid:
            dx = best_dx + ddx
            dy = best_dy + ddy
            center = (float(center_com["cx"]) + dx, float(center_com["cy"]) + dy)
            dp_crop = crop_for_center(center, size=192)
            tag = f"dx{dx:+d}_dy{dy:+d}"
            out_dir = diag_root / tag
            cfg = build_single_slice_cfg(base, rotation_deg=theta0, iters=iters, out_dir=out_dir, dp_crop=dp_crop)
            cfg_path = out_dir / "config.yaml"
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg.to_yaml(cfg_path)
            jobs.append(
                Job(
                    name=tag,
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
                        f"fine_center_{tag}",
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
        "theta0_deg": theta0,
        "iters": iters,
        "seed": seed,
        "center_base": {"cx": center_base[0], "cy": center_base[1]},
    }
    (diag_root / "job_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

