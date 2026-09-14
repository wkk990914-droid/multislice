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
    out_root = base_run / "diagnostics" / "rotation_refine"
    out_root.mkdir(parents=True, exist_ok=True)

    base = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    base.validate()

    best_center_path = base_run / "diagnostics" / "BEST_DP_CENTER.json"
    if not best_center_path.exists():
        raise SystemExit(f"missing {best_center_path}; run center sweep first")
    best_center_payload = json.loads(best_center_path.read_text(encoding="utf-8"))
    best_center = (
        float(best_center_payload["BEST_DP_CENTER_Q256"]["cx"]),
        float(best_center_payload["BEST_DP_CENTER_Q256"]["cy"]),
    )
    dp_crop = tuple(int(v) for v in best_center_payload["BEST_DP_CROP"])

    theta0 = 17.0
    iters = 18
    seed = 0
    gpus = [5, 6, 7]

    angles = [theta0 + d for d in (-3, -2, -1, 0, 1, 2, 3)]
    jobs: list[Job] = []
    for ang in angles:
        tag = f"theta{ang:+.2f}".replace("+", "p").replace("-", "m")
        out_dir = out_root / tag
        cfg = build_single_slice_cfg(base, rotation_deg=ang, iters=iters, out_dir=out_dir, dp_crop=dp_crop)
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
                    f"rot_refine_{ang:+g}",
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

    scheduler = GpuScheduler(pool, log_path=out_root / "gpu_assignment.log", env_builder=env_builder)
    result = scheduler.run(jobs)

    payload = {
        "ok": bool(result.ok),
        "aborted": result.aborted,
        "theta0_deg": theta0,
        "angles": angles,
        "iters": iters,
        "seed": seed,
        "best_center": {"cx": best_center[0], "cy": best_center[1]},
        "dp_crop": list(int(v) for v in dp_crop),
    }
    (out_root / "job_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
