from __future__ import annotations

import copy
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ptycho.config import PipelineConfig, StageConfig
from ptycho.gpu_scheduler import GpuPool, GpuScheduler, Job

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_ROOT
    / "PTYREX_8_FOV5.0nm_def0nm_alpha25.0mrad_dx0.020nm_R256x256_Q256x256_CL0.00082m_ro73deg_trueCL87.h5"
)


def build_cfg(
    base: PipelineConfig,
    out_dir: Path,
    rotation_deg: float,
    iters: int,
    transpose_dp: bool,
    flip_dp_x: bool,
    flip_dp_y: bool,
    transpose_scan: bool,
    flip_scan_x: bool,
    flip_scan_y: bool,
) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    cfg.output.dir = str(out_dir)
    cfg.experiment.probe_energy = 300000.0
    cfg.experiment.probe_semiangle = 25.0
    cfg.data.dp_sampling_mrad = 1.2155
    cfg.data.raw_path = str(DATA_PATH)
    cfg.model.num_slices = 1
    cfg.preprocess.force_com_rotation = float(rotation_deg)

    cfg.data.dp_crop = (30, 222, 32, 224)
    cfg.data.scan_crop = (96, 160, 96, 160)

    cfg.data.transpose_dp = bool(transpose_dp)
    cfg.data.flip_dp_x = bool(flip_dp_x)
    cfg.data.flip_dp_y = bool(flip_dp_y)
    cfg.data.transpose_scan = bool(transpose_scan)
    cfg.data.flip_scan_x = bool(flip_scan_x)
    cfg.data.flip_scan_y = bool(flip_scan_y)

    base_stage = base.stage("pixelated_baseline")
    cfg.stages = [
        StageConfig(
            name="single_slice",
            kind="pixelated",
            num_iters=int(iters),
            reset=True,
            batch_size=base_stage.batch_size,
            preprocess_batch_size=base_stage.preprocess_batch_size,
            optimizer=base_stage.optimizer,
            constraints=base_stage.constraints,
        )
    ]
    return cfg


def main() -> int:
    root = PROJECT_ROOT / "runs" / "single_slice_geometry_screening_300kev"
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.json").write_text(
        json.dumps(
            {
                "data": str(DATA_PATH),
                "beam_energy_eV": 300000.0,
                "alpha_mrad": 25.0,
                "dp_sampling_mrad_per_px": 1.2155,
                "dp_crop": [30, 222, 32, 224],
                "scan_crop": [96, 160, 96, 160],
                "note": "All runs are single-slice, fixed dp_sampling (no scan).",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    base = PipelineConfig.from_yaml(PROJECT_ROOT / "configs" / "pyrex_atomic_phase.yaml")
    base.validate()

    jobs: list[Job] = []
    run_specs: list[dict] = []

    baseline_iters = 50
    baseline_rot = 73.0
    axis_iters = 50
    scan_conv_iters = 50

    def add_run(
        group: str,
        name: str,
        rotation: float,
        iters: int,
        tdp: bool = False,
        fx: bool = False,
        fy: bool = False,
        ts: bool = False,
        fsx: bool = False,
        fsy: bool = False,
    ):
        tag = f"{group}__{name}"
        out_dir = root / "runs" / tag
        cfg = build_cfg(
            base,
            out_dir=out_dir,
            rotation_deg=rotation,
            iters=iters,
            transpose_dp=tdp,
            flip_dp_x=fx,
            flip_dp_y=fy,
            transpose_scan=ts,
            flip_scan_x=fsx,
            flip_scan_y=fsy,
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
        run_specs.append(
            {
                "tag": tag,
                "group": group,
                "name": name,
                "rotation_deg": float(rotation),
                "iters": int(iters),
                "transpose_dp": bool(tdp),
                "flip_dp_x": bool(fx),
                "flip_dp_y": bool(fy),
                "transpose_scan": bool(ts),
                "flip_scan_x": bool(fsx),
                "flip_scan_y": bool(fsy),
            }
        )

    add_run("baseline", "rot_73", baseline_rot, baseline_iters)

    rot_angles = list(range(69, 78)) + [-73, 163, -17]
    for ang in rot_angles:
        add_run("rotation", f"rot_{ang:+d}".replace("+", "p").replace("-", "m"), float(ang), 50)

    add_run("axis", "identity", baseline_rot, axis_iters, tdp=False, fx=False, fy=False)
    add_run("axis", "transpose", baseline_rot, axis_iters, tdp=True, fx=False, fy=False)
    add_run("axis", "flip_qx", baseline_rot, axis_iters, tdp=False, fx=True, fy=False)
    add_run("axis", "flip_qy", baseline_rot, axis_iters, tdp=False, fx=False, fy=True)
    add_run("axis", "flip_qx_qy", baseline_rot, axis_iters, tdp=False, fx=True, fy=True)
    add_run("axis", "transpose_flip_qx", baseline_rot, axis_iters, tdp=True, fx=True, fy=False)
    add_run("axis", "transpose_flip_qy", baseline_rot, axis_iters, tdp=True, fx=False, fy=True)

    add_run("scan", "transpose_scan", baseline_rot, scan_conv_iters, ts=True, fsx=False, fsy=False)
    add_run("scan", "flip_scan_x", baseline_rot, scan_conv_iters, ts=False, fsx=True, fsy=False)
    add_run("scan", "flip_scan_y", baseline_rot, scan_conv_iters, ts=False, fsx=False, fsy=True)

    (root / "run_specs.json").write_text(json.dumps(run_specs, indent=2), encoding="utf-8")

    pool = GpuPool(allowed=[5, 6, 7], min_free_mb=12000)

    def env_builder(gpu: int) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "4",
            "PYREX_SEED": "0",
        }

    scheduler = GpuScheduler(pool, log_path=root / "gpu_assignment.log", env_builder=env_builder)
    result = scheduler.run(jobs)

    csv_path = root / "gpu_assignment.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(
            [
                "tag",
                "gpu",
                "returncode",
                "runtime_s",
                "worker_runtime_s",
                "peak_memory_mb",
                "initial_loss",
                "final_loss",
            ]
        )
        for j in jobs:
            metrics_path = root / "runs" / j.name / "stages" / "single_slice" / "metrics.json"
            metrics = None
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            w.writerow(
                [
                    j.name,
                    j.gpu,
                    j.returncode,
                    round(j.runtime, 3),
                    metrics.get("runtime_s") if metrics else None,
                    metrics.get("peak_memory_mb") if metrics else None,
                    metrics.get("initial_loss") if metrics else None,
                    metrics.get("final_loss") if metrics else None,
                ]
            )

    (root / "job_summary.json").write_text(
        json.dumps(
            {
                "ok": bool(result.ok),
                "aborted": result.aborted,
                "n_jobs": len(jobs),
                "jobs": [asdict(j) for j in jobs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
