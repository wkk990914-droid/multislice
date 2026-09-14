from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from ptycho.config import PipelineConfig, StageConfig
from ptycho.gpu_scheduler import GpuPool, GpuScheduler, Job


def build_cfg(
    base: PipelineConfig,
    out_dir: Path,
    rotation_deg: float,
    transpose_dp: bool,
    flip_dp_x: bool,
    flip_dp_y: bool,
    dp_crop: tuple[int, int, int, int],
) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    cfg.output.dir = str(out_dir)
    cfg.model.num_slices = 1
    cfg.data.dp_crop = tuple(int(v) for v in dp_crop)
    cfg.data.transpose_dp = bool(transpose_dp)
    cfg.data.flip_dp_x = bool(flip_dp_x)
    cfg.data.flip_dp_y = bool(flip_dp_y)
    cfg.preprocess.force_com_rotation = float(rotation_deg)
    cfg.stages = [
        StageConfig(
            name="single_slice",
            kind="pixelated",
            num_iters=18,
            reset=True,
            batch_size=64,
            preprocess_batch_size=64,
            optimizer=base.stage("pixelated_baseline").optimizer,
            constraints=base.stage("pixelated_baseline").constraints,
        )
    ]
    return cfg


def main() -> int:
    base_cfg = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    base_cfg.validate()

    src_best = Path("runs/pyrex_atomic_phase_test/diagnostics/BEST_DP_CENTER.json")
    best = json.loads(src_best.read_text(encoding="utf-8"))
    dp_crop = tuple(int(v) for v in best["BEST_DP_CROP"])

    run_root = Path("runs/pyrex_300keV_geometry_validation")
    matrix_root = run_root / "geometry_matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)
    (matrix_root / "started.txt").write_text("started\n", encoding="utf-8")

    conventions = [
        ("A_identity", False, False, False),
        ("B_transpose", True, False, False),
        ("C_flip_x", False, True, False),
        ("D_flip_y", False, False, True),
        ("E_transpose_flip_x", True, True, False),
        ("F_transpose_flip_y", True, False, True),
        ("G_flip_xy", False, True, True),
        ("H_transpose_flip_xy", True, True, True),
    ]
    rotations = [73.0, 17.0, -73.0, -17.0]

    seed = 0
    jobs: list[Job] = []
    for conv, tdp, fx, fy in conventions:
        for rot in rotations:
            tag = f"{conv}__rot_{rot:+.0f}".replace("+", "p").replace("-", "m")
            out_dir = matrix_root / tag
            cfg = build_cfg(
                base_cfg,
                out_dir=out_dir,
                rotation_deg=rot,
                transpose_dp=tdp,
                flip_dp_x=fx,
                flip_dp_y=fy,
                dp_crop=dp_crop,
            )
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
                        tag,
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

    scheduler = GpuScheduler(pool, log_path=matrix_root / "gpu_assignment.log", env_builder=env_builder)
    result = scheduler.run(jobs)
    (matrix_root / "job_summary.json").write_text(
        json.dumps(
            {
                "ok": bool(result.ok),
                "aborted": result.aborted,
                "seed": seed,
                "dp_crop": list(dp_crop),
                "probe_energy_eV": float(base_cfg.experiment.probe_energy),
                "jobs": [
                    {"name": j.name, "gpu": j.gpu, "returncode": j.returncode, "runtime_s": j.runtime}
                    for j in jobs
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
