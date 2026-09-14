from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

from ptycho.config import PipelineConfig, StageConfig
from ptycho.gpu_scheduler import GpuPool, GpuScheduler, Job


def robust_clim(data: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    lo, hi = np.nanpercentile(np.asarray(data, dtype=np.float64), [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


def fft_power(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    img = img - np.nanmean(img)
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(img * win))
    return np.abs(F) ** 2


def peak_to_background(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=np.float64)
    ny, nx = p.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core = r < max(4.0, 0.05 * min(ny, nx))
    ring = (~core) & (r < 0.45 * min(ny, nx))
    bg = float(np.median(p[ring])) if ring.any() else float(np.median(p))
    pk = float(np.max(p[ring])) if ring.any() else float(np.max(p))
    return float(pk / bg) if bg > 0 else 0.0


def tile_uniformity(image: np.ndarray, tiles: int = 4) -> dict[str, float]:
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    ty = ny // tiles
    tx = nx // tiles
    p2b = []
    for iy in range(tiles):
        for ix in range(tiles):
            tile = img[iy * ty : (iy + 1) * ty, ix * tx : (ix + 1) * tx]
            p2b.append(peak_to_background(fft_power(tile)))
    p2b = np.asarray(p2b, dtype=np.float64)
    coverage = float((p2b > 10.0).sum() / p2b.size)
    return {
        "coverage": coverage,
        "median": float(np.median(p2b)),
        "min": float(np.min(p2b)),
        "max": float(np.max(p2b)),
    }


def build_cfg(
    base: PipelineConfig,
    out_dir: Path,
    dp_sampling_mrad: float,
    dp_crop_size: int,
    rotation_deg: float,
    transpose_dp: bool,
    flip_dp_x: bool,
    flip_dp_y: bool,
    dp_center_px: tuple[float, float],
) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    cfg.output.dir = str(out_dir)
    cfg.model.num_slices = 1
    cfg.experiment.probe_energy = 300000.0
    cfg.data.dp_sampling_mrad = float(dp_sampling_mrad)
    cfg.data.transpose_dp = bool(transpose_dp)
    cfg.data.flip_dp_x = bool(flip_dp_x)
    cfg.data.flip_dp_y = bool(flip_dp_y)

    half = dp_crop_size / 2.0
    cx, cy = dp_center_px
    qx0 = int(np.round(cx - half))
    qy0 = int(np.round(cy - half))
    cfg.data.dp_crop = (qy0, qy0 + dp_crop_size, qx0, qx0 + dp_crop_size)

    cfg.data.scan_crop = (0, 64, 0, 64)
    cfg.preprocess.force_com_rotation = float(rotation_deg)
    cfg.stages = [
        StageConfig(
            name="single_slice",
            kind="pixelated",
            num_iters=40,
            reset=True,
            batch_size=64,
            preprocess_batch_size=64,
            optimizer=base.stage("pixelated_baseline").optimizer,
            constraints=base.stage("pixelated_baseline").constraints,
        )
    ]
    return cfg


def render_outputs(run_dir: Path, out_dir: Path, tag: str) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj = np.load(run_dir / "stages" / "single_slice" / "obj_cropped.npy")
    loss = np.load(run_dir / "stages" / "single_slice" / "loss_history.npy").astype(np.float64)
    img = np.asarray(obj[0], dtype=np.float64)

    vmin, vmax = robust_clim(img, 0.5, 99.5)
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300)
    ax.imshow(img, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"{tag}\\nphase/potential", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}__phase.png", bbox_inches="tight")
    plt.close(fig)

    trans = np.exp(1j * img)
    amp = np.abs(trans)
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300)
    ax.imshow(amp, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_title(f"{tag}\\namplitude(|exp(i phase)|)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}__amplitude.png", bbox_inches="tight")
    plt.close(fig)

    p = fft_power(img)
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300)
    ax.imshow(np.log1p(p), cmap="inferno", origin="lower")
    ax.set_title(f"{tag}\\nFFT log(1+|F|^2)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}__phase_fft.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=300)
    ax.plot(loss, lw=1.1)
    ax.set_title(f"{tag} loss", fontsize=9)
    ax.set_xlabel("iter")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}__loss_curve.png", bbox_inches="tight")
    plt.close(fig)

    u = tile_uniformity(img, tiles=4)
    p2b = peak_to_background(p)
    return {
        "final_loss": float(loss[-1]),
        "initial_loss": float(loss[0]),
        "loss_reduction_percent": float(100.0 * (loss[0] - loss[-1]) / loss[0]) if loss[0] != 0 else 0.0,
        "phase_std": float(np.std(img)),
        "fft_peak_prominence": float(p2b),
        "tile_coverage": float(u["coverage"]),
        "tile_median_peak_bg": float(u["median"]),
        "tile_min_peak_bg": float(u["min"]),
    }


def main() -> int:
    out_root = Path("runs/dp_sampling_audit_300kev")
    audit = json.loads((out_root / "dp_sampling_audit.json").read_text(encoding="utf-8"))
    dp_center = audit["centers_px"]["fit"]
    dp_center_px = (float(dp_center[0]), float(dp_center[1]))
    prev_center_px = None
    prev_best = Path("runs/pyrex_atomic_phase_test/diagnostics/BEST_DP_CENTER.json")
    if prev_best.exists():
        payload = json.loads(prev_best.read_text(encoding="utf-8"))
        prev_center = payload.get("BEST_DP_CENTER_Q256")
        if prev_center:
            prev_center_px = (float(prev_center["cx"]), float(prev_center["cy"]))

    dp_sampling_cal = audit["dp_sampling_calibrated_mrad_per_px"]
    dp_sampling_meta = audit["dp_sampling_metadata_mrad_per_px"] or dp_sampling_cal
    dp_crop_rec = int(audit["recommended_dp_crop"] or 192)

    base = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    base.validate()

    matrix = [
        ("meta", float(dp_sampling_meta), 192, 73.0),
        ("cal", float(dp_sampling_cal), 192, 73.0),
    ]
    if prev_center_px is not None:
        matrix.append(("cal_prevcenter", float(dp_sampling_cal), 192, 73.0))
    if dp_crop_rec != 192:
        matrix.append(("cal_recrop", float(dp_sampling_cal), dp_crop_rec, 73.0))

    for d in (-2.0, -1.0, 0.0, 1.0, 2.0):
        matrix.append((f"rot{d:+.0f}".replace("+", "p").replace("-", "m"), float(dp_sampling_cal), dp_crop_rec, 73.0 + d))

    runs_dir = out_root / "screening_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = []
    for run_id, dp_s, crop_sz, rot in matrix:
        tag = f"{run_id}__dp{dp_s:.6f}__N{crop_sz}__rot{rot:+.2f}".replace("+", "p").replace("-", "m")
        run_dir = runs_dir / tag
        center_use = prev_center_px if run_id == "cal_prevcenter" and prev_center_px is not None else dp_center_px
        cfg = build_cfg(
            base,
            out_dir=run_dir,
            dp_sampling_mrad=dp_s,
            dp_crop_size=int(crop_sz),
            rotation_deg=float(rot),
            transpose_dp=False,
            flip_dp_x=False,
            flip_dp_y=False,
            dp_center_px=center_use,
        )
        cfg_path = run_dir / "config.yaml"
        run_dir.mkdir(parents=True, exist_ok=True)
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
                    str(run_dir),
                    "--gpu-label",
                    tag,
                ],
                deps=[],
                log_path=run_dir / "worker.log",
            )
        )

    pool = GpuPool(allowed=[5, 6, 7], min_free_mb=8000)

    def env_builder(gpu: int) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "4",
            "PYREX_SEED": "0",
        }

    scheduler = GpuScheduler(pool, log_path=out_root / "gpu_assignment.log", env_builder=env_builder)
    result = scheduler.run(jobs)

    out_img_dir = out_root / "screening_outputs"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for j in jobs:
        run_dir = runs_dir / j.name
        if not (run_dir / "stages" / "single_slice" / "loss_history.npy").exists():
            rows.append(
                {
                    "run_id": j.name,
                    "GPU": j.gpu,
                    "status": "MISSING_OUTPUT",
                }
            )
            continue
        metrics = render_outputs(run_dir, out_img_dir, j.name)
        rows.append(
            {
                "run_id": j.name,
                "GPU": j.gpu,
                "beam_energy": 300000.0,
                "alpha": 25.0,
                "dp_sampling": float(run_dir.name.split("__dp")[1].split("__")[0]),
                "dp_crop": int(run_dir.name.split("__N")[1].split("__")[0]),
                "rotation": float(run_dir.name.split("__rot")[1].replace("p", "+").replace("m", "-")),
                "initial_loss": metrics["initial_loss"],
                "final_loss": metrics["final_loss"],
                "loss_reduction_percent": metrics["loss_reduction_percent"],
                "phase_std": metrics["phase_std"],
                "atomic_periodicity_score": metrics["tile_coverage"],
                "fft_peak_prominence": metrics["fft_peak_prominence"],
                "tile_min_peak_bg": metrics["tile_min_peak_bg"],
                "runtime": j.runtime,
                "status": "OK" if j.returncode == 0 else f"RC{j.returncode}",
            }
        )

    csv_path = out_root / "screening_summary.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    (out_root / "job_summary.json").write_text(
        json.dumps(
            {
                "ok": bool(result.ok),
                "aborted": result.aborted,
                "n_jobs": len(jobs),
                "dp_center_px": list(dp_center_px),
                "recommended_dp_crop": dp_crop_rec,
                "dp_sampling_cal": dp_sampling_cal,
                "dp_sampling_meta": dp_sampling_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
