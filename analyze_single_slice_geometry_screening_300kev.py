from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def robust_clim_stack(arrays: list[np.ndarray], low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    stack = np.stack([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
    lo, hi = np.nanpercentile(stack, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


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


def peak_to_background(power: np.ndarray) -> float:
    ratio, _pk, _bg = peak_to_background_stats(power)
    return ratio


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


def region_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    y0 = int(round(0.25 * ny))
    y1 = int(round(0.75 * ny))
    x0 = int(round(0.25 * nx))
    x1 = int(round(0.75 * nx))
    center = np.zeros((ny, nx), dtype=bool)
    center[y0:y1, x0:x1] = True
    edge = ~center
    return center, edge


def periodicity_score(image: np.ndarray) -> float:
    return peak_to_background(fft_power(image))


def finite_ok(arr: np.ndarray) -> bool:
    a = np.asarray(arr)
    return bool(np.isfinite(a).all())


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def rank_key(row: "Row") -> tuple[float, float, float, float, float, float]:
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


@dataclass
class Row:
    tag: str
    group: str
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
    periodicity_detected: bool
    finite: bool
    peak_memory_mb: float | None
    runtime_s: float | None


def save_panel(image: np.ndarray, path: Path, title: str, vmin: float, vmax: float, cmap: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300)
    ax.plot(loss, lw=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("iter")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def montage(tags: list[str], arrays: dict[str, np.ndarray], out_path: Path, vmin: float, vmax: float, title: str) -> None:
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
            ax.imshow(arrays[tag], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(tag.replace("rotation__", "").replace("axis__", ""), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = PROJECT_ROOT / "runs" / "single_slice_geometry_screening_300kev"
    run_specs = json.loads((root / "run_specs.json").read_text(encoding="utf-8"))

    run_root = root / "runs"
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_arrays: dict[str, np.ndarray] = {}
    fft_arrays: dict[str, np.ndarray] = {}
    rows: list[Row] = []
    peaks_out: dict[str, list] = {}

    for spec in run_specs:
        tag = spec["tag"]
        stage_dir = run_root / tag / "stages" / "single_slice"
        if not stage_dir.exists():
            continue
        obj_path = stage_dir / "obj_cropped.npy"
        loss_path = stage_dir / "loss_history.npy"
        metrics_path = stage_dir / "metrics.json"
        if not (obj_path.exists() and loss_path.exists() and metrics_path.exists()):
            continue
        obj = np.load(obj_path)
        probe_path = stage_dir / "probe.npy"
        loss = np.load(loss_path).astype(np.float64)
        if loss.size == 0:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        phase = np.asarray(obj[0], dtype=np.float64)

        phase_arrays[tag] = phase
        power = fft_power(phase)
        fft_arrays[tag] = np.log1p(power)

        center_mask, edge_mask = region_masks(phase.shape)
        ny, nx = phase.shape
        cy, cx = ny // 2, nx // 2
        y0 = int(round(0.25 * ny))
        y1 = int(round(0.75 * ny))
        x0 = int(round(0.25 * nx))
        x1 = int(round(0.75 * nx))
        center_score = periodicity_score(phase[y0:y1, x0:x1])
        edge_score = periodicity_score(np.where(edge_mask, phase, 0.0))

        p2b, pk, bg = peak_to_background_stats(power)
        sym = symmetry_score(power)
        hf = high_freq_fraction(power)
        finite = finite_ok(phase) and finite_ok(loss)

        periodicity_detected = bool(p2b > 10.0 and sym >= 0.3)
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
        peaks_out[tag] = {
            "peak_bg_ratio": float(p2b),
            "bg_median": float(bg),
            "ring_peak": float(pk),
            "peaks": peaks,
        }

        rows.append(
            Row(
                tag=tag,
                group=spec["group"],
                rotation_deg=float(spec["rotation_deg"]),
                transpose_dp=bool(spec["transpose_dp"]),
                flip_dp_x=bool(spec["flip_dp_x"]),
                flip_dp_y=bool(spec["flip_dp_y"]),
                transpose_scan=bool(spec["transpose_scan"]),
                flip_scan_x=bool(spec["flip_scan_x"]),
                flip_scan_y=bool(spec["flip_scan_y"]),
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
                periodicity_detected=periodicity_detected,
                finite=finite,
                peak_memory_mb=float(metrics.get("peak_memory_mb")) if metrics.get("peak_memory_mb") is not None else None,
                runtime_s=float(metrics.get("runtime_s")) if metrics.get("runtime_s") is not None else None,
            )
        )

        run_out = out_dir / "per_run" / tag
        run_out.mkdir(parents=True, exist_ok=True)
        np.save(run_out / "final_phase.npy", phase.astype(np.float32))
        np.save(run_out / "final_fft_log.npy", fft_arrays[tag].astype(np.float32))

        vmin, vmax = float(np.percentile(phase, 0.5)), float(np.percentile(phase, 99.5))
        final_loss = safe_float(loss[-1])
        save_panel(
            phase,
            run_out / "phase.png",
            f"{tag}\\nrot={spec['rotation_deg']} loss={final_loss:.3g}",
            vmin,
            vmax,
            "viridis",
        )
        trans = np.exp(1j * (phase - float(np.mean(phase))))
        amp = np.abs(trans)
        save_panel(
            amp,
            run_out / "amplitude.png",
            f"{tag}\\n|exp(i·phase)|",
            0.0,
            1.0,
            "gray",
        )
        save_panel(
            np.log1p(power),
            run_out / "phase_fft.png",
            f"{tag}\\npeak/bg={p2b:.2f} sym={sym:.2f}",
            float(np.min(np.log1p(power))),
            float(np.max(np.log1p(power))),
            "inferno",
        )
        if probe_path.exists():
            probe = np.load(probe_path)
            p0 = probe[0] if probe.ndim == 3 else probe
            pamp = np.abs(p0)
            ppha = np.angle(p0)
            save_panel(
                np.log1p(pamp),
                run_out / "probe_amplitude.png",
                f"{tag}\\nprobe log(1+|P|)",
                float(np.min(np.log1p(pamp))),
                float(np.max(np.log1p(pamp))),
                "inferno",
            )
            save_panel(
                ppha,
                run_out / "probe_phase.png",
                f"{tag}\\nprobe phase",
                float(np.percentile(ppha, 0.5)),
                float(np.percentile(ppha, 99.5)),
                "twilight",
            )
        save_loss(loss, run_out / "loss_curve.png", tag)

    if not rows:
        raise SystemExit("no runs found")

    phase_vmin, phase_vmax = robust_clim_stack(list(phase_arrays.values()), 0.5, 99.5)
    fft_vmin, fft_vmax = robust_clim_stack(list(fft_arrays.values()), 0.5, 99.5)

    rot_tags = [r.tag for r in rows if r.group == "rotation"]
    rot_tags = sorted(rot_tags, key=lambda t: float(t.split("rot_")[-1].replace("p", "").replace("m", "-")) if "rot_" in t else 0.0)
    if rot_tags:
        montage(rot_tags, phase_arrays, out_dir / "FINAL_rotation_screening_phase_montage.png", phase_vmin, phase_vmax, "rotation screening — phase")
        montage(rot_tags, fft_arrays, out_dir / "FINAL_rotation_screening_fft_montage.png", fft_vmin, fft_vmax, "rotation screening — FFT")

    axis_tags = [r.tag for r in rows if r.group == "axis"]
    if axis_tags:
        montage(axis_tags, phase_arrays, out_dir / "FINAL_axis_screening_phase_montage.png", phase_vmin, phase_vmax, "axis convention screening — phase")
        montage(axis_tags, fft_arrays, out_dir / "FINAL_axis_screening_fft_montage.png", fft_vmin, fft_vmax, "axis convention screening — FFT")

    csv_path = out_dir / "screening_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow([f.name for f in Row.__dataclass_fields__.values()])
        for r in rows:
            w.writerow([getattr(r, f.name) for f in Row.__dataclass_fields__.values()])

    (out_dir / "fft_peaks.json").write_text(json.dumps(peaks_out, indent=2), encoding="utf-8")

    ranked = sorted(rows, key=rank_key)
    (out_dir / "ranked_runs.json").write_text(
        json.dumps([r.__dict__ for r in ranked], indent=2), encoding="utf-8"
    )
    axis_ranked = [r for r in ranked if r.group == "axis"]
    if axis_ranked:
        (out_dir / "top_axis_candidates.json").write_text(
            json.dumps([r.__dict__ for r in axis_ranked[:3]], indent=2),
            encoding="utf-8",
        )
    best = ranked[0]
    (out_dir / "best_run.json").write_text(json.dumps(best.__dict__, indent=2), encoding="utf-8")

    best_cfg = PipelineConfig.from_yaml(run_root / best.tag / "config.yaml")
    best_cfg.to_yaml(root / "best_single_slice_geometry.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
