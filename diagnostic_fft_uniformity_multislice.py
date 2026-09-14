from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fft_power(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    img = img - np.nanmean(img)
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(img * win))
    return np.abs(F) ** 2


def fft_peak_to_bg(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=np.float64)
    ny, nx = p.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core = r < max(4.0, 0.05 * min(ny, nx))
    ring = (~core) & (r < 0.45 * min(ny, nx))
    bg = float(np.median(p[ring]))
    pk = float(np.max(p[ring]))
    return float(pk / bg) if bg > 0 else 0.0


def spatial_uniformity(image: np.ndarray, tiles: int = 4) -> tuple[np.ndarray, dict]:
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    ty = ny // tiles
    tx = nx // tiles
    metrics = np.zeros((tiles, tiles), dtype=np.float64)
    for iy in range(tiles):
        for ix in range(tiles):
            tile = img[iy * ty : (iy + 1) * ty, ix * tx : (ix + 1) * tx]
            metrics[iy, ix] = fft_peak_to_bg(fft_power(tile))
    summary = {
        "tiles": tiles,
        "median_peak_to_background": float(np.median(metrics)),
        "min_peak_to_background": float(np.min(metrics)),
        "max_peak_to_background": float(np.max(metrics)),
    }
    return metrics, summary


def plot(metrics: np.ndarray, path: Path, title: str) -> None:
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


def main() -> int:
    run_dir = Path("runs/pyrex_atomic_phase_test")
    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = np.load(run_dir / "stages" / "dip_free_extra" / "obj_cropped.npy")
    proj = np.asarray(obj, dtype=np.float64).sum(axis=0)
    metrics, summary = spatial_uniformity(proj, tiles=4)
    plot(metrics, out_dir / "fft_spatial_uniformity.png", "multislice projected: spatial FFT uniformity (4x4)")
    (out_dir / "fft_spatial_uniformity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
