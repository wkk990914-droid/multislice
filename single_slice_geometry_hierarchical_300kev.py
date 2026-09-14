from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ptycho.config import PipelineConfig, StageConfig
from ptycho.gpu_scheduler import GpuPool, GpuScheduler, Job

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_ROOT
    / "PTYREX_8_FOV5.0nm_def0nm_alpha25.0mrad_dx0.020nm_R256x256_Q256x256_CL0.00082m_ro73deg_trueCL87.h5"
)
ROOT = PROJECT_ROOT / "runs" / "single_slice_geometry_hierarchical_300kev"

FIXED_ENERGY_EV = 300000.0
FIXED_ALPHA_MRAD = 25.0
FIXED_DP_SAMPLING_MRAD_PER_PX = 1.2155
FIXED_NUM_SLICES = 1
FIXED_SCAN_CROP = (96, 160, 96, 160)
FIXED_DP_CROP = (30, 222, 32, 224)
ALLOWED_GPUS = (5, 6, 7)


@dataclass
class RunSpec:
    tag: str
    stage: str
    rotation_deg: float
    iters: int
    transpose_dp: bool
    flip_dp_x: bool
    flip_dp_y: bool
    transpose_scan: bool
    flip_scan_x: bool
    flip_scan_y: bool


@dataclass
class Row:
    tag: str
    stage: str
    rotation_deg: float
    transpose_dp: bool
    flip_dp_x: bool
    flip_dp_y: bool
    transpose_scan: bool
    flip_scan_x: bool
    flip_scan_y: bool
    initial_loss: float
    final_loss: float
    loss_reduction_percent: float
    phase_std: float
    phase_hf_power_frac: float
    fft_peak_prominence: float
    fft_symmetry: float
    center_periodicity: float
    edge_periodicity: float
    center_edge_periodicity_ratio: float
    periodicity_detected: bool
    finite: bool
    peak_memory_mb: float | None
    runtime_s: float | None


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def fft_power(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    img = img - np.nanmean(img)
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(img * win))
    return np.abs(F) ** 2


def high_freq_fraction(power: np.ndarray, r0_frac: float = 0.25) -> float:
    p = np.asarray(power, dtype=np.float64)
    ny, nx = p.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = r >= r0_frac * min(ny, nx)
    tot = float(p.sum())
    return float(p[mask].sum() / tot) if tot > 0 else 0.0


def peak_list(power: np.ndarray, n: int = 20, r_min: float = 6.0) -> list[tuple[int, int, float]]:
    p = np.asarray(power, dtype=np.float64)
    ny, nx = p.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    work = np.where(r >= r_min, p, -np.inf)
    flat = work.ravel()
    if n >= flat.size:
        idx = np.argsort(flat)[::-1]
    else:
        idx = np.argpartition(flat, -n)[-n:]
        idx = idx[np.argsort(flat[idx])[::-1]]
    out = []
    for i in idx:
        if not np.isfinite(flat[i]):
            continue
        y = int(i // nx)
        x = int(i % nx)
        out.append((y, x, float(p[y, x])))
    return out


def peak_to_background_stats(power: np.ndarray) -> tuple[float, float, float]:
    p = np.asarray(power, dtype=np.float64)
    ny, nx = p.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core = r < max(4.0, 0.05 * min(ny, nx))
    ring = (~core) & (r < 0.45 * min(ny, nx))
    bg = float(np.median(p[ring])) if ring.any() else float(np.median(p))
    pk = float(np.max(p[ring])) if ring.any() else float(np.max(p))
    ratio = float(pk / bg) if bg > 0 else 0.0
    return ratio, pk, bg


def symmetry_score(power: np.ndarray, tol_px: float = 2.0) -> float:
    peaks = peak_list(power, n=20, r_min=8.0)
    if not peaks:
        return 0.0
    p = np.asarray(power, dtype=np.float64)
    ny, nx = p.shape
    cy, cx = ny // 2, nx // 2
    matched = 0
    for y, x, v in peaks[:10]:
        my = int(round(2 * cy - y))
        mx = int(round(2 * cx - x))
        y0 = max(0, int(my - tol_px))
        y1 = min(ny, int(my + tol_px) + 1)
        x0 = max(0, int(mx - tol_px))
        x1 = min(nx, int(mx + tol_px) + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        vv = float(np.max(p[y0:y1, x0:x1]))
        if vv > 0.2 * v:
            matched += 1
    return float(matched / min(10, len(peaks)))


def finite_ok(arr: np.ndarray) -> bool:
    a = np.asarray(arr)
    return bool(np.isfinite(a).all())


def robust_clim_stack(
    arrays: list[np.ndarray], low: float = 0.5, high: float = 99.5
) -> tuple[float, float]:
    stack = np.stack([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
    lo, hi = np.nanpercentile(stack, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


def periodicity_score(image: np.ndarray) -> float:
    ratio, _pk, _bg = peak_to_background_stats(fft_power(image))
    return ratio


def rank_key(row: Row) -> tuple[float, float, float, float, float, float]:
    center = safe_float(row.center_periodicity, default=float("-inf"))
    sym = safe_float(row.fft_symmetry, default=float("-inf"))
    prom = safe_float(row.fft_peak_prominence, default=float("-inf"))
    red = safe_float(row.loss_reduction_percent, default=float("-inf"))
    final = safe_float(row.final_loss, default=float("inf"))
    return (
        -(1.0 if bool(row.periodicity_detected) else 0.0),
        -center,
        -sym,
        -prom,
        -red,
        final,
    )


def normalised_scores(rows: list[Row]) -> dict[str, float]:
    def col(name: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            v = safe_float(getattr(r, name))
            if np.isfinite(v):
                out.append(v)
        return out

    def minmax(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 1.0
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax <= vmin:
            vmax = vmin + 1e-12
        return vmin, vmax

    mm = {
        "center_periodicity": minmax(col("center_periodicity")),
        "fft_symmetry": minmax(col("fft_symmetry")),
        "fft_peak_prominence": minmax(col("fft_peak_prominence")),
        "loss_reduction_percent": minmax(col("loss_reduction_percent")),
        "final_loss": minmax(col("final_loss")),
    }

    scores: dict[str, float] = {}
    for r in rows:
        det = 1.0 if r.periodicity_detected else 0.0
        def scale(name: str, inv: bool = False) -> float:
            v = safe_float(getattr(r, name))
            vmin, vmax = mm[name]
            x = (v - vmin) / (vmax - vmin) if np.isfinite(v) else 0.0
            x = float(np.clip(x, 0.0, 1.0))
            return 1.0 - x if inv else x

        score = (
            100.0 * det
            + 10.0 * scale("center_periodicity")
            + 5.0 * scale("fft_symmetry")
            + 3.0 * scale("fft_peak_prominence")
            + 2.0 * scale("loss_reduction_percent")
            + 1.0 * scale("final_loss", inv=True)
        )
        scores[r.tag] = float(score)
    return scores


def save_panel(image: np.ndarray, path: Path, title: str, vmin: float, vmax: float, cmap: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300)
    ax.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_loss(loss: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300)
    ax.plot(loss, lw=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("iter")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def montage(
    tags: list[str],
    arrays: dict[str, np.ndarray],
    out_path: Path,
    vmin: float,
    vmax: float,
    title: str,
    cmap: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(tags)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows), dpi=300)
    axes = np.asarray(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            tag = tags[i]
            ax.imshow(arrays[tag], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(tag, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def copy_alias(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


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
    reset: bool,
) -> PipelineConfig:
    cfg = PipelineConfig.from_dict(base.to_dict())
    cfg.output.dir = str(out_dir)
    cfg.experiment.probe_energy = FIXED_ENERGY_EV
    cfg.experiment.probe_semiangle = FIXED_ALPHA_MRAD
    cfg.data.dp_sampling_mrad = FIXED_DP_SAMPLING_MRAD_PER_PX
    cfg.data.raw_path = str(DATA_PATH)
    cfg.model.num_slices = FIXED_NUM_SLICES
    cfg.preprocess.force_com_rotation = float(rotation_deg)

    cfg.data.dp_crop = FIXED_DP_CROP
    cfg.data.scan_crop = FIXED_SCAN_CROP

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
            reset=bool(reset),
            batch_size=base_stage.batch_size,
            preprocess_batch_size=base_stage.preprocess_batch_size,
            optimizer=base_stage.optimizer,
            constraints=base_stage.constraints,
        )
    ]
    return cfg


def env_builder(gpu: int) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "4",
        "PYREX_SEED": "0",
    }


def write_run_specs(path: Path, specs: list[RunSpec]) -> None:
    payload = [asdict(s) for s in specs]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_jobs(base: PipelineConfig, specs: list[RunSpec]) -> list[Job]:
    jobs: list[Job] = []
    for s in specs:
        out_dir = ROOT / "runs" / s.tag
        cfg = build_cfg(
            base,
            out_dir=out_dir,
            rotation_deg=s.rotation_deg,
            iters=s.iters,
            transpose_dp=s.transpose_dp,
            flip_dp_x=s.flip_dp_x,
            flip_dp_y=s.flip_dp_y,
            transpose_scan=s.transpose_scan,
            flip_scan_x=s.flip_scan_x,
            flip_scan_y=s.flip_scan_y,
            reset=True,
        )
        cfg_path = out_dir / "config.yaml"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(cfg_path)
        jobs.append(
            Job(
                name=s.tag,
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
                    s.tag,
                ],
                deps=[],
                log_path=out_dir / "worker.log",
            )
        )
    return jobs


def ensure_root() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "README.json").write_text(
        json.dumps(
            {
                "data": str(DATA_PATH),
                "beam_energy_eV": FIXED_ENERGY_EV,
                "alpha_mrad": FIXED_ALPHA_MRAD,
                "dp_sampling_mrad_per_px": FIXED_DP_SAMPLING_MRAD_PER_PX,
                "num_slices": FIXED_NUM_SLICES,
                "dp_crop": list(FIXED_DP_CROP),
                "scan_crop": list(FIXED_SCAN_CROP),
                "gpus": list(ALLOWED_GPUS),
                "note": "Hierarchical single-slice screening; no dp_sampling sweep; no multislice.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ensure_gpu_assignment_header()


def ensure_gpu_assignment_header() -> None:
    csv_path = ROOT / "gpu_assignment.csv"
    if csv_path.exists():
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(
            [
                "stage",
                "tag",
                "gpu",
                "start_time",
                "end_time",
                "runtime_s",
                "worker_runtime_s",
                "peak_memory_mb",
                "returncode",
                "status",
            ]
        )


def append_gpu_assignment(stage: str, jobs: list[Job]) -> None:
    csv_path = ROOT / "gpu_assignment.csv"
    ensure_gpu_assignment_header()
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        for j in jobs:
            metrics_path = ROOT / "runs" / j.name / "stages" / "single_slice" / "metrics.json"
            metrics = None
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            start_s = datetime.fromtimestamp(j.start).strftime("%Y-%m-%d %H:%M:%S") if j.start else None
            end_s = datetime.fromtimestamp(j.end).strftime("%Y-%m-%d %H:%M:%S") if j.end else None
            ok = j.returncode == 0
            w.writerow(
                [
                    stage,
                    j.name,
                    j.gpu,
                    start_s,
                    end_s,
                    round(j.runtime, 3),
                    metrics.get("runtime_s") if metrics else None,
                    metrics.get("peak_memory_mb") if metrics else None,
                    j.returncode,
                    "ok" if ok else "failed",
                ]
            )


def analyze_stage(
    stage: str,
    specs: list[RunSpec],
    out_prefix: str,
) -> dict[str, Any]:
    run_root = ROOT / "runs"

    rows: list[Row] = []
    phase_arrays: dict[str, np.ndarray] = {}
    fft_arrays: dict[str, np.ndarray] = {}
    peaks_out: dict[str, Any] = {}

    for s in specs:
        stage_dir = run_root / s.tag / "stages" / "single_slice"
        if not stage_dir.exists():
            continue
        obj_path = stage_dir / "obj_cropped.npy"
        loss_path = stage_dir / "loss_history.npy"
        metrics_path = stage_dir / "metrics.json"
        if not (obj_path.exists() and loss_path.exists() and metrics_path.exists()):
            continue
        obj = np.load(obj_path)
        loss = np.load(loss_path).astype(np.float64)
        if loss.size == 0:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        phase = np.asarray(obj[0], dtype=np.float64)

        power = fft_power(phase)
        fft_log = np.log1p(power)

        ny, nx = phase.shape
        y0 = int(round(0.25 * ny))
        y1 = int(round(0.75 * ny))
        x0 = int(round(0.25 * nx))
        x1 = int(round(0.75 * nx))
        center_phase = phase[y0:y1, x0:x1]
        center_score = periodicity_score(center_phase)

        edge_mask = np.ones((ny, nx), dtype=bool)
        edge_mask[y0:y1, x0:x1] = False
        edge_score = periodicity_score(np.where(edge_mask, phase, 0.0))

        ratio_center_edge = safe_float(center_score / edge_score) if edge_score and np.isfinite(edge_score) else float("nan")

        p2b, pk, bg = peak_to_background_stats(power)
        sym = symmetry_score(power)
        hf = high_freq_fraction(power)
        finite = finite_ok(phase) and finite_ok(loss)

        periodicity_detected = bool(p2b > 10.0 and sym >= 0.3)

        cy, cx = ny // 2, nx // 2
        peaks: list[dict] = []
        for y, x, v in peak_list(power, n=20, r_min=8.0):
            dy = float(y - cy)
            dx = float(x - cx)
            r = float(np.sqrt(dy * dy + dx * dx))
            my = int(round(2 * cy - y))
            mx = int(round(2 * cx - x))
            y0m = max(0, my - 2)
            y1m = min(ny, my + 3)
            x0m = max(0, mx - 2)
            x1m = min(nx, mx + 3)
            sv = float(np.max(power[y0m:y1m, x0m:x1m])) if (y1m > y0m and x1m > x0m) else 0.0
            peaks.append(
                {
                    "y": int(y),
                    "x": int(x),
                    "dy": dy,
                    "dx": dx,
                    "r_px": r,
                    "value": float(v),
                    "value_over_bg": float(v / bg) if bg > 0 else 0.0,
                    "sym_y": int(my),
                    "sym_x": int(mx),
                    "sym_value": sv,
                    "sym_value_over_bg": float(sv / bg) if bg > 0 else 0.0,
                }
            )
        peaks_out[s.tag] = {
            "peak_bg_ratio": float(p2b),
            "bg_median": float(bg),
            "ring_peak": float(pk),
            "peaks": peaks,
        }

        row = Row(
            tag=s.tag,
            stage=stage,
            rotation_deg=float(s.rotation_deg),
            transpose_dp=bool(s.transpose_dp),
            flip_dp_x=bool(s.flip_dp_x),
            flip_dp_y=bool(s.flip_dp_y),
            transpose_scan=bool(s.transpose_scan),
            flip_scan_x=bool(s.flip_scan_x),
            flip_scan_y=bool(s.flip_scan_y),
            initial_loss=safe_float(loss[0]),
            final_loss=safe_float(loss[-1]),
            loss_reduction_percent=(
                safe_float(100.0 * (loss[0] - loss[-1]) / loss[0])
                if loss[0] != 0 and np.isfinite(loss[0]) and np.isfinite(loss[-1])
                else float("nan")
            ),
            phase_std=safe_float(np.std(phase)),
            phase_hf_power_frac=safe_float(hf),
            fft_peak_prominence=safe_float(p2b),
            fft_symmetry=safe_float(sym),
            center_periodicity=safe_float(center_score),
            edge_periodicity=safe_float(edge_score),
            center_edge_periodicity_ratio=safe_float(ratio_center_edge),
            periodicity_detected=periodicity_detected,
            finite=bool(finite),
            peak_memory_mb=safe_float(metrics.get("peak_memory_mb")) if metrics.get("peak_memory_mb") is not None else None,
            runtime_s=safe_float(metrics.get("runtime_s")) if metrics.get("runtime_s") is not None else None,
        )
        rows.append(row)
        phase_arrays[s.tag] = phase
        fft_arrays[s.tag] = fft_log

    if not rows:
        raise SystemExit(f"{stage}: no completed runs found under {run_root}")

    scores = normalised_scores(rows)
    ranked = sorted(rows, key=rank_key)
    ranked_payload = []
    for r in ranked:
        d = dict(r.__dict__)
        d["composite_score"] = scores.get(r.tag)
        ranked_payload.append(d)

    csv_path = ROOT / f"{out_prefix}_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow([f.name for f in Row.__dataclass_fields__.values()])
        for r in rows:
            w.writerow([getattr(r, f.name) for f in Row.__dataclass_fields__.values()])

    (ROOT / f"{out_prefix}_ranked.json").write_text(
        json.dumps(ranked_payload, indent=2), encoding="utf-8"
    )
    (ROOT / f"{out_prefix}_fft_peaks.json").write_text(
        json.dumps(peaks_out, indent=2), encoding="utf-8"
    )

    phase_vmin, phase_vmax = robust_clim_stack(list(phase_arrays.values()), 0.5, 99.5)
    fft_vmin, fft_vmax = robust_clim_stack(list(fft_arrays.values()), 0.5, 99.5)

    tags = [r.tag for r in ranked]
    montage(
        tags,
        phase_arrays,
        ROOT / f"{out_prefix}_phase_montage.png",
        phase_vmin,
        phase_vmax,
        f"{stage} — phase",
        cmap="viridis",
    )
    montage(
        tags,
        fft_arrays,
        ROOT / f"{out_prefix}_fft_montage.png",
        fft_vmin,
        fft_vmax,
        f"{stage} — FFT",
        cmap="inferno",
    )

    best = ranked[0]
    top2 = ranked[:2] if len(ranked) >= 2 else ranked[:1]
    score_best = safe_float(scores.get(best.tag), default=float("nan"))
    close = False
    if len(top2) >= 2:
        score_2 = safe_float(scores.get(top2[1].tag), default=float("nan"))
        if np.isfinite(score_best) and score_best > 0 and np.isfinite(score_2):
            diff_pct = 100.0 * abs(score_best - score_2) / score_best
            close = diff_pct < 10.0

    selection = {
        "stage": stage,
        "timestamp": now_stamp(),
        "best_tag": best.tag,
        "best_rotation_deg": best.rotation_deg,
        "top2_tags": [r.tag for r in top2],
        "top2_close": bool(close),
        "ranking_priority": [
            "periodicity_detected",
            "center_periodicity",
            "fft_symmetry",
            "fft_peak_prominence",
            "loss_reduction_percent",
            "final_loss",
        ],
    }
    (ROOT / f"{out_prefix}_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    return selection


def stage_a_specs() -> list[RunSpec]:
    angles = list(range(69, 78)) + [-73, 163, -17]
    out: list[RunSpec] = []
    seen = set()
    for ang in angles:
        tag = f"stageA__rot_{ang:+d}".replace("+", "p").replace("-", "m")
        if tag in seen:
            continue
        seen.add(tag)
        out.append(
            RunSpec(
                tag=tag,
                stage="A",
                rotation_deg=float(ang),
                iters=50,
                transpose_dp=False,
                flip_dp_x=False,
                flip_dp_y=False,
                transpose_scan=False,
                flip_scan_x=False,
                flip_scan_y=False,
            )
        )
    return out


def stage_b_specs(rotation_degs: list[float]) -> list[RunSpec]:
    variants = [
        ("identity", False, False, False),
        ("transpose", True, False, False),
        ("flip_qx", False, True, False),
        ("flip_qy", False, False, True),
        ("flip_qx_qy", False, True, True),
        ("transpose_flip_qx", True, True, False),
        ("transpose_flip_qy", True, False, True),
    ]
    out: list[RunSpec] = []
    for rot in rotation_degs:
        rot_tag = f"rot_{int(round(rot)):+d}".replace("+", "p").replace("-", "m")
        for name, tdp, fx, fy in variants:
            tag = f"stageB__{rot_tag}__{name}"
            out.append(
                RunSpec(
                    tag=tag,
                    stage="B",
                    rotation_deg=float(rot),
                    iters=50,
                    transpose_dp=bool(tdp),
                    flip_dp_x=bool(fx),
                    flip_dp_y=bool(fy),
                    transpose_scan=False,
                    flip_scan_x=False,
                    flip_scan_y=False,
                )
            )
    return out


def stage_c_specs(candidates: list[dict[str, Any]]) -> list[RunSpec]:
    variants = [
        ("identity", False, False, False),
        ("transpose_scan", True, False, False),
        ("flip_scan_x", False, True, False),
        ("flip_scan_y", False, False, True),
        ("flip_scan_x_y", False, True, True),
        ("transpose_flip_scan_x", True, True, False),
        ("transpose_flip_scan_y", True, False, True),
    ]
    out: list[RunSpec] = []
    for cand in candidates:
        rot = float(cand["rotation_deg"])
        tdp = bool(cand["transpose_dp"])
        fx = bool(cand["flip_dp_x"])
        fy = bool(cand["flip_dp_y"])
        rot_tag = f"rot_{int(round(rot)):+d}".replace("+", "p").replace("-", "m")
        dp_tag = "dpT" if tdp else "dpI"
        dp_tag += ("_fx" if fx else "")
        dp_tag += ("_fy" if fy else "")
        base = f"stageC__{rot_tag}__{dp_tag}"
        for name, ts, fsx, fsy in variants:
            tag = f"{base}__{name}"
            out.append(
                RunSpec(
                    tag=tag,
                    stage="C",
                    rotation_deg=float(rot),
                    iters=50,
                    transpose_dp=tdp,
                    flip_dp_x=fx,
                    flip_dp_y=fy,
                    transpose_scan=bool(ts),
                    flip_scan_x=bool(fsx),
                    flip_scan_y=bool(fsy),
                )
            )
    return out


def read_ranked(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_top_geometry(out_prefix: str) -> list[dict[str, Any]]:
    ranked = read_ranked(ROOT / f"{out_prefix}_ranked.json")
    if not ranked:
        raise SystemExit(f"{out_prefix}: ranked is empty")
    best = ranked[0]
    top2 = ranked[:2] if len(ranked) >= 2 else ranked[:1]
    score_best = safe_float(best.get("composite_score"), default=float("nan"))
    close = False
    if len(top2) >= 2:
        score_2 = safe_float(top2[1].get("composite_score"), default=float("nan"))
        if np.isfinite(score_best) and score_best > 0 and np.isfinite(score_2):
            diff_pct = 100.0 * abs(score_best - score_2) / score_best
            close = diff_pct < 10.0
    return top2 if close else [best]


def run_scheduler(stage: str, jobs: list[Job], min_free_mb: int = 12000) -> None:
    pool = GpuPool(allowed=list(ALLOWED_GPUS), min_free_mb=min_free_mb)
    scheduler = GpuScheduler(pool, log_path=ROOT / "gpu_assignment.log", env_builder=env_builder)
    result = scheduler.run(jobs)
    append_gpu_assignment(stage, jobs)
    (ROOT / f"stage{stage}_job_summary.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "timestamp": now_stamp(),
                "ok": bool(result.ok),
                "aborted": result.aborted,
                "n_jobs": len(jobs),
                "jobs": [asdict(j) for j in jobs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not result.ok:
        raise SystemExit(f"Stage {stage}: scheduler aborted={result.aborted}")


def os_env() -> dict[str, str]:
    import os

    return os.environ.copy()


def pick_gpu(min_free_mb: int = 12000) -> int:
    pool = GpuPool(allowed=list(ALLOWED_GPUS), min_free_mb=min_free_mb)
    avail = pool.available()
    if not avail:
        raise SystemExit("No GPU available under the current min_free_mb constraint.")
    return int(avail[0].index)


def stage_d_convergence(best_tag: str) -> None:
    run_root = ROOT / "runs"
    best_cfg_path = run_root / best_tag / "config.yaml"
    if not best_cfg_path.exists():
        raise SystemExit(f"Stage D: best config not found: {best_cfg_path}")
    base = PipelineConfig.from_yaml(PROJECT_ROOT / "configs" / "pyrex_atomic_phase.yaml")
    base.validate()

    best_cfg = PipelineConfig.from_yaml(best_cfg_path)
    rot = float(best_cfg.preprocess.force_com_rotation or 0.0)
    tdp = bool(best_cfg.data.transpose_dp)
    fx = bool(best_cfg.data.flip_dp_x)
    fy = bool(best_cfg.data.flip_dp_y)
    ts = bool(best_cfg.data.transpose_scan)
    fsx = bool(best_cfg.data.flip_scan_x)
    fsy = bool(best_cfg.data.flip_scan_y)

    out_dir = ROOT / "runs" / "stageD__convergence_best"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = pick_gpu(min_free_mb=16000)

    cfg_050 = build_cfg(
        base,
        out_dir=out_dir,
        rotation_deg=rot,
        iters=50,
        transpose_dp=tdp,
        flip_dp_x=fx,
        flip_dp_y=fy,
        transpose_scan=ts,
        flip_scan_x=fsx,
        flip_scan_y=fsy,
        reset=True,
    )
    cfg_100 = build_cfg(
        base,
        out_dir=out_dir,
        rotation_deg=rot,
        iters=50,
        transpose_dp=tdp,
        flip_dp_x=fx,
        flip_dp_y=fy,
        transpose_scan=ts,
        flip_scan_x=fsx,
        flip_scan_y=fsy,
        reset=False,
    )
    cfg_200 = build_cfg(
        base,
        out_dir=out_dir,
        rotation_deg=rot,
        iters=100,
        transpose_dp=tdp,
        flip_dp_x=fx,
        flip_dp_y=fy,
        transpose_scan=ts,
        flip_scan_x=fsx,
        flip_scan_y=fsy,
        reset=False,
    )

    cfg_050_path = out_dir / "config_iter050.yaml"
    cfg_100_path = out_dir / "config_iter100.yaml"
    cfg_200_path = out_dir / "config_iter200.yaml"
    cfg_050.to_yaml(cfg_050_path)
    cfg_100.to_yaml(cfg_100_path)
    cfg_200.to_yaml(cfg_200_path)

    log_path = out_dir / "stageD_worker.log"
    log_path.write_text("", encoding="utf-8")

    def run_one(cfg_path: Path, continue_from_self: bool, label: str) -> None:
        started = time.time()
        argv = [
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
            label,
        ]
        if continue_from_self:
            argv.append("--continue-from-self")
        env = os_env()
        env.update(env_builder(gpu))
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"[{now_stamp()}] RUN {' '.join(argv)}\n")
        subprocess.run(argv, check=True, env=env)
        ended = time.time()
        stage_dir = out_dir / "stages" / "single_slice"
        metrics_path = stage_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        csv_path = ROOT / "gpu_assignment.csv"
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(
                [
                    "D",
                    label,
                    gpu,
                    datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime.fromtimestamp(ended).strftime("%Y-%m-%d %H:%M:%S"),
                    round(ended - started, 3),
                    metrics.get("runtime_s"),
                    metrics.get("peak_memory_mb"),
                    0,
                    "ok",
                ]
            )

    run_one(cfg_050_path, continue_from_self=False, label="stageD_iter050")
    export_convergence_snapshot(out_dir, suffix="050")
    run_one(cfg_100_path, continue_from_self=True, label="stageD_iter100")
    export_convergence_snapshot(out_dir, suffix="100")
    run_one(cfg_200_path, continue_from_self=True, label="stageD_iter200")
    export_convergence_snapshot(out_dir, suffix="200")
    export_convergence_summary(out_dir)


def export_convergence_snapshot(out_dir: Path, suffix: str) -> None:
    stage_dir = out_dir / "stages" / "single_slice"
    obj_path = stage_dir / "obj_cropped.npy"
    loss_path = stage_dir / "loss_history.npy"
    if not (obj_path.exists() and loss_path.exists()):
        raise SystemExit(f"Stage D snapshot missing: {stage_dir}")
    obj = np.load(obj_path)
    loss = np.load(loss_path).astype(np.float64)
    phase = np.asarray(obj[0], dtype=np.float64)
    power = fft_power(phase)
    fft_log = np.log1p(power)

    conv_dir = ROOT / "convergence_arrays"
    conv_dir.mkdir(parents=True, exist_ok=True)
    np.save(conv_dir / f"phase_iter{suffix}.npy", phase.astype(np.float32))
    np.save(conv_dir / f"fft_log_iter{suffix}.npy", fft_log.astype(np.float32))
    np.save(conv_dir / f"loss_iter{suffix}.npy", loss.astype(np.float64))

    ny, nx = phase.shape
    y0 = int(round(0.25 * ny))
    y1 = int(round(0.75 * ny))
    x0 = int(round(0.25 * nx))
    x1 = int(round(0.75 * nx))
    center = phase[y0:y1, x0:x1]
    center_score = periodicity_score(center)
    edge_mask = np.ones((ny, nx), dtype=bool)
    edge_mask[y0:y1, x0:x1] = False
    edge_score = periodicity_score(np.where(edge_mask, phase, 0.0))
    ratio_center_edge = (
        safe_float(center_score / edge_score) if edge_score and np.isfinite(edge_score) else float("nan")
    )
    p2b, _pk, _bg = peak_to_background_stats(power)
    sym = symmetry_score(power)

    metrics_path = ROOT / "convergence_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    payload[suffix] = {
        "iterations": int(suffix),
        "initial_loss": safe_float(loss[0]) if loss.size else None,
        "final_loss": safe_float(loss[-1]) if loss.size else None,
        "center_periodicity": safe_float(center_score),
        "edge_periodicity": safe_float(edge_score),
        "center_edge_periodicity_ratio": safe_float(ratio_center_edge),
        "fft_peak_prominence": safe_float(p2b),
        "fft_symmetry": safe_float(sym),
        "finite": bool(finite_ok(phase) and finite_ok(loss)),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_convergence_summary(out_dir: Path) -> None:
    conv_dir = ROOT / "convergence_arrays"
    p050 = conv_dir / "phase_iter050.npy"
    p100 = conv_dir / "phase_iter100.npy"
    p200 = conv_dir / "phase_iter200.npy"
    f050 = conv_dir / "fft_log_iter050.npy"
    f100 = conv_dir / "fft_log_iter100.npy"
    f200 = conv_dir / "fft_log_iter200.npy"
    l200 = conv_dir / "loss_iter200.npy"

    if not (p050.exists() and p100.exists() and p200.exists() and f050.exists() and f100.exists() and f200.exists() and l200.exists()):
        raise SystemExit("Stage D summary missing convergence_arrays/*.npy")

    phase_stack = [np.load(p).astype(np.float64) for p in (p050, p100, p200)]
    fft_stack = [np.load(p).astype(np.float64) for p in (f050, f100, f200)]
    phase_vmin, phase_vmax = robust_clim_stack(phase_stack, 0.5, 99.5)
    fft_vmin, fft_vmax = robust_clim_stack(fft_stack, 0.5, 99.5)

    save_panel(phase_stack[0], ROOT / "phase_iter050.png", "iter050", phase_vmin, phase_vmax, "viridis")
    save_panel(phase_stack[1], ROOT / "phase_iter100.png", "iter100", phase_vmin, phase_vmax, "viridis")
    save_panel(phase_stack[2], ROOT / "phase_iter200.png", "iter200", phase_vmin, phase_vmax, "viridis")
    save_panel(fft_stack[0], ROOT / "fft_iter050.png", "iter050", fft_vmin, fft_vmax, "inferno")
    save_panel(fft_stack[1], ROOT / "fft_iter100.png", "iter100", fft_vmin, fft_vmax, "inferno")
    save_panel(fft_stack[2], ROOT / "fft_iter200.png", "iter200", fft_vmin, fft_vmax, "inferno")

    copy_alias(ROOT / "phase_iter200.png", ROOT / "FINAL_best_phase.png")
    copy_alias(ROOT / "fft_iter200.png", ROOT / "FINAL_best_phase_fft.png")

    loss = np.load(l200).astype(np.float64)
    save_loss(loss, ROOT / "loss_000_200.png", "loss 0-200 (cumulative)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=300)
    for ax, arr, name in zip(axes, phase_stack, ["050", "100", "200"]):
        ax.imshow(arr, cmap="viridis", origin="lower", vmin=phase_vmin, vmax=phase_vmax)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "FINAL_convergence_phase_050_100_200.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=300)
    for ax, arr, name in zip(axes, fft_stack, ["050", "100", "200"]):
        ax.imshow(arr, cmap="inferno", origin="lower", vmin=fft_vmin, vmax=fft_vmax)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "FINAL_convergence_fft_050_100_200.png", bbox_inches="tight")
    plt.close(fig)


def stage_f_summary() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Stage A", ROOT / "STAGE_A_rotation_phase_montage.png"),
        ("Stage B", ROOT / "STAGE_B_axis_phase_montage.png"),
        ("Stage C", ROOT / "STAGE_C_scan_phase_montage.png"),
        ("Convergence", ROOT / "FINAL_convergence_phase_050_100_200.png"),
    ]
    images = []
    titles = []
    for t, p in panels:
        if p.exists():
            images.append(plt.imread(p))
            titles.append(t)
        else:
            images.append(None)
            titles.append(t)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), dpi=300)
    axes = np.asarray(axes).reshape(2, 2)
    for ax, im, title in zip(axes.ravel(), images, titles):
        if im is None:
            ax.axis("off")
            ax.set_title(title, fontsize=11)
            continue
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "FINAL_single_slice_geometry_summary.png", bbox_inches="tight")
    plt.close(fig)


def write_best_yaml(best_tag: str) -> None:
    cfg_path = ROOT / "runs" / best_tag / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Best config not found: {cfg_path}")
    cfg = PipelineConfig.from_yaml(cfg_path)
    cfg.to_yaml(ROOT / "best_single_slice_geometry.yaml")


def write_final_report() -> None:
    def load_json(path: Path) -> Any:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    a_sel = load_json(ROOT / "stageA_selection.json") or {}
    a_rank = load_json(ROOT / "stageA_ranked.json") or []
    b_rank = load_json(ROOT / "stageB_ranked.json") or []
    c_rank = load_json(ROOT / "stageC_ranked.json") or []
    conv = load_json(ROOT / "convergence_metrics.json") or {}

    def by_tag(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for it in items:
            if isinstance(it, dict) and "tag" in it:
                out[str(it["tag"])] = it
        return out

    a_map = by_tag(a_rank)
    stageA_best = str(a_sel.get("best_tag")) if a_sel.get("best_tag") else (a_rank[0]["tag"] if a_rank else None)
    stageA_top2 = list(a_sel.get("top2_tags", [])) if a_sel.get("top2_tags") else ([r["tag"] for r in a_rank[:2]] if a_rank else [])

    def rotation_from_tag(tag: str) -> float | None:
        it = a_map.get(tag)
        if it is None:
            return None
        return safe_float(it.get("rotation_deg"))

    rot_best = rotation_from_tag(stageA_best) if stageA_best else None
    rot_2 = rotation_from_tag(stageA_top2[1]) if len(stageA_top2) >= 2 else None

    def metric(tag: str, key: str) -> float | None:
        it = a_map.get(tag)
        if it is None:
            return None
        v = safe_float(it.get(key))
        return v if np.isfinite(v) else None

    diff_pct = None
    if len(stageA_top2) >= 2:
        s1 = safe_float(a_map.get(stageA_top2[0], {}).get("composite_score"))
        s2 = safe_float(a_map.get(stageA_top2[1], {}).get("composite_score"))
        if np.isfinite(s1) and s1 > 0 and np.isfinite(s2):
            diff_pct = float(100.0 * abs(s1 - s2) / s1)

    def find_by_rotation(deg: int) -> dict[str, Any] | None:
        for it in a_rank:
            if int(round(safe_float(it.get("rotation_deg"), default=9999))) == int(deg):
                return it
        return None

    rot_p73 = find_by_rotation(73)
    rot_m73 = find_by_rotation(-73)
    rot_p163 = find_by_rotation(163)
    rot_m17 = find_by_rotation(-17)

    b_best = b_rank[0] if b_rank else None
    c_best = c_rank[0] if c_rank else None

    best_yaml = ROOT / "best_single_slice_geometry.yaml"
    cfg = PipelineConfig.from_yaml(best_yaml) if best_yaml.exists() else None

    report: dict[str, Any] = {
        "timestamp": now_stamp(),
        "fixed": {
            "beam_energy_eV": FIXED_ENERGY_EV,
            "alpha_mrad": FIXED_ALPHA_MRAD,
            "dp_sampling_mrad_per_px": FIXED_DP_SAMPLING_MRAD_PER_PX,
            "num_slices": FIXED_NUM_SLICES,
            "scan_crop": list(FIXED_SCAN_CROP),
            "dp_crop": list(FIXED_DP_CROP),
            "gpus": list(ALLOWED_GPUS),
        },
        "stageA": {
            "best_rotation_deg": rot_best,
            "second_rotation_deg": rot_2,
            "top2_composite_diff_percent": diff_pct,
            "rotation_sign_test": {
                "+73": rot_p73,
                "-73": rot_m73,
            },
            "equivalent_orientation_test": {
                "+163": rot_p163,
                "-17": rot_m17,
            },
        },
        "stageB": {
            "best_candidate": b_best,
            "best_dp_axis_convention": {
                "transpose_dp": bool(b_best.get("transpose_dp")) if b_best else None,
                "flip_dp_x": bool(b_best.get("flip_dp_x")) if b_best else None,
                "flip_dp_y": bool(b_best.get("flip_dp_y")) if b_best else None,
            }
            if b_best
            else None,
        },
        "stageC": {
            "best_candidate": c_best,
            "best_scan_convention": {
                "transpose_scan": bool(c_best.get("transpose_scan")) if c_best else None,
                "flip_scan_x": bool(c_best.get("flip_scan_x")) if c_best else None,
                "flip_scan_y": bool(c_best.get("flip_scan_y")) if c_best else None,
            }
            if c_best
            else None,
        },
        "best_geometry_yaml": best_yaml.as_posix() if best_yaml.exists() else None,
        "convergence": conv,
    }

    if cfg is not None:
        report["best_geometry_compact"] = {
            "rotation_deg": float(cfg.preprocess.force_com_rotation or 0.0),
            "transpose_dp": bool(cfg.data.transpose_dp),
            "flip_dp_x": bool(cfg.data.flip_dp_x),
            "flip_dp_y": bool(cfg.data.flip_dp_y),
            "transpose_scan": bool(cfg.data.transpose_scan),
            "flip_scan_x": bool(cfg.data.flip_scan_x),
            "flip_scan_y": bool(cfg.data.flip_scan_y),
        }

    report_path = ROOT / "FINAL_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines: list[str] = []
    md_lines.append("# SINGLE_SLICE_GEOMETRY hierarchical summary")
    md_lines.append("")
    md_lines.append(f"- timestamp: {report['timestamp']}")
    md_lines.append(f"- fixed: energy={FIXED_ENERGY_EV/1000:.0f} keV, alpha={FIXED_ALPHA_MRAD} mrad, dp_sampling={FIXED_DP_SAMPLING_MRAD_PER_PX} mrad/px, single-slice")
    md_lines.append("")
    md_lines.append("## Stage A (rotation)")
    md_lines.append(f"- best_rotation_deg: {rot_best}")
    md_lines.append(f"- second_rotation_deg: {rot_2}")
    md_lines.append(f"- top2_composite_diff_percent: {diff_pct}")
    md_lines.append("")
    md_lines.append("## Stage B (diffraction-axis)")
    if b_best:
        md_lines.append(f"- best_tag: {b_best.get('tag')}")
        md_lines.append(f"- rotation_deg: {b_best.get('rotation_deg')}")
        md_lines.append(f"- transpose_dp: {b_best.get('transpose_dp')}, flip_dp_x: {b_best.get('flip_dp_x')}, flip_dp_y: {b_best.get('flip_dp_y')}")
    else:
        md_lines.append("- best_tag: null")
    md_lines.append("")
    md_lines.append("## Stage C (scan-axis)")
    if c_best:
        md_lines.append(f"- best_tag: {c_best.get('tag')}")
        md_lines.append(f"- rotation_deg: {c_best.get('rotation_deg')}")
        md_lines.append(f"- transpose_scan: {c_best.get('transpose_scan')}, flip_scan_x: {c_best.get('flip_scan_x')}, flip_scan_y: {c_best.get('flip_scan_y')}")
    else:
        md_lines.append("- best_tag: null")
    md_lines.append("")
    md_lines.append("## Convergence (50/100/200)")
    for k in ["050", "100", "200"]:
        it = conv.get(k)
        if it is None:
            continue
        md_lines.append(
            f"- iter{k}: loss={it.get('final_loss')}, center_periodicity={it.get('center_periodicity')}, "
            f"edge_periodicity={it.get('edge_periodicity')}, ratio={it.get('center_edge_periodicity_ratio')}, "
            f"sym={it.get('fft_symmetry')}, peak/bg={it.get('fft_peak_prominence')}, finite={it.get('finite')}"
        )
    md_lines.append("")
    md_lines.append("## Gate")
    md_lines.append("- SINGLE_SLICE_GEOMETRY must be decided using BOTH real-space phase images and reciprocal-space FFT evidence.")
    (ROOT / "FINAL_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B", "C", "D", "F"], required=True)
    args = parser.parse_args()

    ensure_root()
    base = PipelineConfig.from_yaml(PROJECT_ROOT / "configs" / "pyrex_atomic_phase.yaml")
    base.validate()
    print(f"[hierarchical] stage={args.stage} root={ROOT}", flush=True)

    if args.stage == "A":
        print("[hierarchical] STAGE A: rotation screening (identity axes)", flush=True)
        specs = stage_a_specs()
        write_run_specs(ROOT / "stageA_run_specs.json", specs)
        jobs = make_jobs(base, specs)
        run_scheduler("A", jobs)
        analyze_stage("STAGE A — Rotation screening", specs, out_prefix="stageA_rotation")
        copy_alias(ROOT / "stageA_rotation_ranked.json", ROOT / "stageA_ranked.json")
        copy_alias(
            ROOT / "stageA_rotation_phase_montage.png",
            ROOT / "STAGE_A_rotation_phase_montage.png",
        )
        copy_alias(
            ROOT / "stageA_rotation_fft_montage.png",
            ROOT / "STAGE_A_rotation_fft_montage.png",
        )
        copy_alias(ROOT / "stageA_rotation_selection.json", ROOT / "stageA_selection.json")
        print("[hierarchical] STAGE A done", flush=True)
        return 0

    if args.stage == "B":
        print("[hierarchical] STAGE B: diffraction-axis screening (rotation from Stage A)", flush=True)
        sel_path = ROOT / "stageA_selection.json"
        if not sel_path.exists():
            raise SystemExit("Stage B requires Stage A selection json. Run --stage A first.")
        selection = json.loads(sel_path.read_text(encoding="utf-8"))
        top2_tags = list(selection.get("top2_tags", []))
        top2_close = bool(selection.get("top2_close", False))
        run_specs_a = json.loads((ROOT / "stageA_run_specs.json").read_text(encoding="utf-8"))
        tag_to_rot = {d["tag"]: float(d["rotation_deg"]) for d in run_specs_a}
        rots = [tag_to_rot[t] for t in top2_tags if t in tag_to_rot]
        if not rots:
            raise SystemExit("Stage B: failed to map Stage A tags to rotations.")
        rots = rots[:2] if top2_close else rots[:1]
        specs = stage_b_specs(rots)
        write_run_specs(ROOT / "stageB_run_specs.json", specs)
        jobs = make_jobs(base, specs)
        run_scheduler("B", jobs)
        analyze_stage("STAGE B — Diffraction-axis screening", specs, out_prefix="stageB_axis")
        copy_alias(ROOT / "stageB_axis_ranked.json", ROOT / "stageB_ranked.json")
        copy_alias(
            ROOT / "stageB_axis_phase_montage.png",
            ROOT / "STAGE_B_axis_phase_montage.png",
        )
        copy_alias(
            ROOT / "stageB_axis_fft_montage.png",
            ROOT / "STAGE_B_axis_fft_montage.png",
        )
        copy_alias(ROOT / "stageB_axis_selection.json", ROOT / "stageB_selection.json")
        print("[hierarchical] STAGE B done", flush=True)
        return 0

    if args.stage == "C":
        print("[hierarchical] STAGE C: scan-axis screening (geometry from Stage B)", flush=True)
        ranked_path = ROOT / "stageB_axis_ranked.json"
        if not ranked_path.exists():
            raise SystemExit("Stage C requires Stage B ranked json. Run --stage B first.")
        candidates = select_top_geometry("stageB_axis")
        specs = stage_c_specs(candidates)
        write_run_specs(ROOT / "stageC_run_specs.json", specs)
        jobs = make_jobs(base, specs)
        run_scheduler("C", jobs)
        analyze_stage("STAGE C — Scan-axis screening", specs, out_prefix="stageC_scan")
        copy_alias(ROOT / "stageC_scan_ranked.json", ROOT / "stageC_ranked.json")
        copy_alias(
            ROOT / "stageC_scan_phase_montage.png",
            ROOT / "STAGE_C_scan_phase_montage.png",
        )
        copy_alias(
            ROOT / "stageC_scan_fft_montage.png",
            ROOT / "STAGE_C_scan_fft_montage.png",
        )
        best = read_ranked(ROOT / "stageC_scan_ranked.json")[0]
        best_tag = str(best["tag"])
        write_best_yaml(best_tag)
        (ROOT / "stageC_best_tag.json").write_text(
            json.dumps({"best_tag": best_tag, "timestamp": now_stamp()}, indent=2),
            encoding="utf-8",
        )
        print("[hierarchical] STAGE C done", flush=True)
        return 0

    if args.stage == "D":
        print("[hierarchical] STAGE D: convergence 50→100→200 (checkpoint continuation)", flush=True)
        best_path = ROOT / "stageC_best_tag.json"
        if not best_path.exists():
            raise SystemExit("Stage D requires Stage C best tag json. Run --stage C first.")
        best_tag = json.loads(best_path.read_text(encoding="utf-8"))["best_tag"]
        stage_d_convergence(best_tag)
        print("[hierarchical] STAGE D done", flush=True)
        return 0

    if args.stage == "F":
        print("[hierarchical] STAGE F: final summary + report", flush=True)
        stage_f_summary()
        write_final_report()
        print("[hierarchical] STAGE F done", flush=True)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
