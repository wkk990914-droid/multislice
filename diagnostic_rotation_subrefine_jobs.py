from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

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


def main() -> int:
    base_run = Path("runs/pyrex_atomic_phase_test")
    out_root = base_run / "diagnostics" / "rotation_subrefine"
    out_root.mkdir(parents=True, exist_ok=True)

    base = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    base.validate()

    best_center_payload = json.loads(
        (base_run / "diagnostics" / "BEST_DP_CENTER.json").read_text(encoding="utf-8")
    )
    dp_crop = tuple(int(v) for v in best_center_payload["BEST_DP_CROP"])

    theta0 = 17.0
    angles = [theta0 - 1.0, theta0 - 0.5, theta0, theta0 + 0.5, theta0 + 1.0]
    iters = 18
    seed = 0

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
                    f"rot_subrefine_{ang:+g}",
                ],
                deps=[],
                log_path=out_dir / "worker.log",
            )
        )

    pool = GpuPool(allowed=[5, 6, 7], min_free_mb=8000)

    def env_builder(gpu: int) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "4",
            "PYREX_SEED": str(seed),
        }

    scheduler = GpuScheduler(pool, log_path=out_root / "gpu_assignment.log", env_builder=env_builder)
    result = scheduler.run(jobs)
    (out_root / "job_summary.json").write_text(
        json.dumps({"ok": bool(result.ok), "aborted": result.aborted, "angles": angles, "iters": iters}, indent=2),
        encoding="utf-8",
    )
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

