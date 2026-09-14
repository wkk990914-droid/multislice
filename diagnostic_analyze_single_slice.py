from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def robust_clim(data: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    lo, hi = np.nanpercentile(np.asarray(data, dtype=np.float64), [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


def save_single(
    data: np.ndarray,
    path: Path,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 280,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=dpi)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_loss(loss: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=250)
    ax.plot(np.arange(1, loss.size + 1), loss, lw=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("single-slice loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fft_power(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    img = img - np.nanmean(img)
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(img * win))
    return np.abs(F) ** 2


def fft_peak_metrics(power: np.ndarray) -> dict[str, float | None]:
    p = np.asarray(power, dtype=np.float64)
    ny, nx = p.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core = r < max(4.0, 0.05 * min(ny, nx))
    ring = (~core) & (r < 0.45 * min(ny, nx))
    if not ring.any():
        return {"peak_to_background": None, "dominant_r_px": None, "dominant_angle_deg": None, "high_freq_fraction": None}
    bg = float(np.median(p[ring]))
    pk = float(np.max(p[ring]))
    idx = np.argmax(np.where(ring, p, -np.inf))
    iy, ix = np.unravel_index(idx, p.shape)
    ang = float(np.degrees(np.arctan2(iy - cy, ix - cx)) % 360)
    dom_r = float(r[iy, ix])
    total = float(p.sum())
    high = (r > 0.25 * min(ny, nx)) & (r < 0.5 * min(ny, nx))
    hf = float(p[high].sum() / total) if total > 0 and high.any() else None
    return {
        "peak_to_background": float(pk / bg) if bg > 0 else None,
        "dominant_r_px": dom_r,
        "dominant_angle_deg": ang,
        "high_freq_fraction": hf,
    }


def spatial_uniformity(image: np.ndarray, tiles: int = 4) -> tuple[np.ndarray, dict[str, float | None]]:
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    ty = ny // tiles
    tx = nx // tiles
    metrics = np.zeros((tiles, tiles), dtype=np.float64)
    dominant_r = np.zeros((tiles, tiles), dtype=np.float64)
    for iy in range(tiles):
        for ix in range(tiles):
            tile = img[iy * ty : (iy + 1) * ty, ix * tx : (ix + 1) * tx]
            p = fft_power(tile)
            m = fft_peak_metrics(p)
            metrics[iy, ix] = m["peak_to_background"] or 0.0
            dominant_r[iy, ix] = m["dominant_r_px"] or 0.0
    summary = {
        "median_peak_to_background": float(np.median(metrics)),
        "min_peak_to_background": float(np.min(metrics)),
        "max_peak_to_background": float(np.max(metrics)),
        "median_dominant_r_px": float(np.median(dominant_r)),
        "dominant_r_px_std": float(np.std(dominant_r)),
    }
    return metrics, summary


def plot_uniformity(metrics: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=260)
    im = ax.imshow(metrics, cmap="magma", origin="lower")
    for (iy, ix), val in np.ndenumerate(metrics):
        ax.text(ix, iy, f"{val:.1f}", ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks(range(metrics.shape[1]))
    ax.set_yticks(range(metrics.shape[0]))
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="FFT peak/background")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@dataclass
class RotationResult:
    tag: str
    rotation_deg: float
    iters: int
    initial_loss: float | None
    final_loss: float | None
    loss_reduction_frac: float | None
    fft_peak_to_bg: float | None
    uniformity_median: float | None
    uniformity_min: float | None


def load_single_slice_result(run_dir: Path, tag: str, rotation_deg: float) -> RotationResult:
    metrics_path = run_dir / "stages" / "single_slice" / "metrics.json"
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    loss = np.load(run_dir / "stages" / "single_slice" / "loss_history.npy").astype(np.float64)
    obj = np.load(run_dir / "stages" / "single_slice" / "obj_cropped.npy")
    img = np.asarray(obj[0], dtype=np.float64)
    power = fft_power(img)
    f = fft_peak_metrics(power)
    u, us = spatial_uniformity(img, tiles=4)
    init = m.get("initial_loss")
    fin = m.get("final_loss")
    red = None
    if init is not None and fin is not None and float(init) != 0:
        red = float((float(init) - float(fin)) / float(init))
    return RotationResult(
        tag=tag,
        rotation_deg=float(rotation_deg),
        iters=int(loss.size),
        initial_loss=float(init) if init is not None else None,
        final_loss=float(fin) if fin is not None else None,
        loss_reduction_frac=red,
        fft_peak_to_bg=f["peak_to_background"],
        uniformity_median=us["median_peak_to_background"],
        uniformity_min=us["min_peak_to_background"],
    )


def main() -> int:
    base_run = Path("runs/pyrex_atomic_phase_test")
    diag_root = base_run / "diagnostics"
    job_root = diag_root / "single_slice_jobs"
    out_dir = diag_root
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = job_root / "baseline_rot_p73_iter50"
    obj = np.load(baseline / "stages" / "single_slice" / "obj_cropped.npy")
    loss = np.load(baseline / "stages" / "single_slice" / "loss_history.npy").astype(np.float64)
    img = np.asarray(obj[0], dtype=np.float64)
    vmin, vmax = robust_clim(img, 0.5, 99.5)
    save_single(
        img,
        out_dir / "single_slice_object.png",
        title="single-slice object (potential/phase, rot=+73 deg, 50 iters)",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        dpi=320,
    )

    p = fft_power(img)
    logp = np.log1p(p)
    save_single(
        logp,
        out_dir / "single_slice_fft.png",
        title="single-slice FFT log(1+|F|^2)",
        cmap="inferno",
        dpi=320,
    )
    save_loss(loss, out_dir / "single_slice_loss.png")

    u, us = spatial_uniformity(img, tiles=4)
    plot_uniformity(u, out_dir / "fft_spatial_uniformity.png", "single-slice spatial FFT uniformity (4x4)")
    (out_dir / "single_slice_uniformity.json").write_text(json.dumps(us, indent=2), encoding="utf-8")

    candidates = [
        ("p73", 73.0),
        ("m73", -73.0),
        ("p17", 17.0),
        ("m17", -17.0),
        ("p163", 163.0),
        ("m163", -163.0),
    ]
    results: list[RotationResult] = []
    for tag, rot in candidates:
        run = job_root / f"rot_{tag}_iter15"
        results.append(load_single_slice_result(run, tag=tag, rotation_deg=rot))

    csv_path = out_dir / "rotation_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(
            [
                "tag",
                "rotation_deg",
                "iters",
                "initial_loss",
                "final_loss",
                "loss_reduction_frac",
                "fft_peak_to_bg",
                "uniformity_median_peak_to_bg",
                "uniformity_min_peak_to_bg",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.tag,
                    f"{r.rotation_deg:g}",
                    r.iters,
                    f"{r.initial_loss:.10g}" if r.initial_loss is not None else "",
                    f"{r.final_loss:.10g}" if r.final_loss is not None else "",
                    f"{r.loss_reduction_frac:.10g}" if r.loss_reduction_frac is not None else "",
                    f"{r.fft_peak_to_bg:.10g}" if r.fft_peak_to_bg is not None else "",
                    f"{r.uniformity_median:.10g}" if r.uniformity_median is not None else "",
                    f"{r.uniformity_min:.10g}" if r.uniformity_min is not None else "",
                ]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [r.tag for r in results]
    x = np.arange(len(results))
    uni = np.array([r.uniformity_median or 0.0 for r in results], dtype=np.float64)
    uni_min = np.array([r.uniformity_min or 0.0 for r in results], dtype=np.float64)
    pk = np.array([r.fft_peak_to_bg or 0.0 for r in results], dtype=np.float64)
    loss_red = np.array([r.loss_reduction_frac or 0.0 for r in results], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=250)
    ax.plot(x, uni, marker="o", label="uniformity median peak/bg")
    ax.plot(x, uni_min, marker="o", label="uniformity min peak/bg")
    ax.plot(x, pk, marker="o", label="global FFT peak/bg")
    ax.plot(x, loss_red, marker="o", label="loss reduction frac")
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("metric (arb.)")
    ax.set_title("rotation convention ranking (single-slice, 15 iterations each)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "rotation_comparison.png", bbox_inches="tight")
    plt.close(fig)

    best = sorted(
        results,
        key=lambda r: (
            -(r.uniformity_median or 0.0),
            -(r.uniformity_min or 0.0),
            -(r.fft_peak_to_bg or 0.0),
        ),
    )
    summary = {
        "best_by_uniformity": [vars(r) for r in best[:3]],
        "all": [vars(r) for r in results],
    }
    (out_dir / "rotation_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
