from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from ptycho.config import PipelineConfig, StageConfig
from ptycho.gpu_scheduler import GpuPool, GpuScheduler, Job


def build_cfg(base: PipelineConfig, out_dir: Path, dp_crop: tuple[int, int, int, int], rotation_deg: float) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    cfg.output.dir = str(out_dir)
    cfg.model.num_slices = 1
    cfg.data.dp_crop = tuple(int(v) for v in dp_crop)
    cfg.preprocess.force_com_rotation = float(rotation_deg)
    cfg.stages = [
        StageConfig(
            name="single_slice",
            kind="pixelated",
            num_iters=50,
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
    out_dir = base_run / "diagnostics" / "best_center_single_slice_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = json.loads((base_run / "diagnostics" / "BEST_DP_CENTER.json").read_text(encoding="utf-8"))
    dp_crop = tuple(int(v) for v in best["BEST_DP_CROP"])
    rotation = 17.0

    base = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    base.validate()
    cfg = build_cfg(base, out_dir=out_dir, dp_crop=dp_crop, rotation_deg=rotation)
    cfg_path = out_dir / "config.yaml"
    cfg.to_yaml(cfg_path)

    job = Job(
        name="best_center_single_slice_50",
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
            "best_center_single_slice_50",
        ],
        deps=[],
        log_path=out_dir / "worker.log",
    )

    pool = GpuPool(allowed=[5, 6, 7], min_free_mb=8000)

    def env_builder(gpu: int) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "4",
            "PYREX_SEED": "0",
        }

    scheduler = GpuScheduler(pool, log_path=out_dir / "gpu_assignment.log", env_builder=env_builder)
    result = scheduler.run([job])
    (out_dir / "job_summary.json").write_text(
        json.dumps(
            {
                "ok": bool(result.ok),
                "aborted": result.aborted,
                "job": {"gpu": job.gpu, "returncode": job.returncode, "runtime_s": job.runtime},
                "dp_crop": list(dp_crop),
                "rotation_deg": rotation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
