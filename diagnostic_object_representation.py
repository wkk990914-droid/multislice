from __future__ import annotations

import json
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
    dpi: int = 250,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=dpi)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_montage(
    stack: np.ndarray,
    path: Path,
    titles: list[str],
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cols: int = 4,
    dpi: int = 300,
    suptitle: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stack = np.asarray(stack, dtype=np.float64)
    nz = int(stack.shape[0])
    rows = int(np.ceil(nz / cols))
    if vmin is None or vmax is None:
        vmin, vmax = robust_clim(stack)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.35 * cols, 2.35 * rows), squeeze=False, dpi=dpi
    )
    flat = axes.ravel()
    last_im = None
    for z in range(nz):
        ax = flat[z]
        last_im = ax.imshow(stack[z], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(titles[z], fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
    for z in range(nz, rows * cols):
        flat[z].axis("off")

    cbar_ax = fig.add_axes([0.925, 0.12, 0.014, 0.74])
    fig.colorbar(last_im, cax=cbar_ax)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=0.985)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def detrend_stack(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    nz, ny, nx = stack.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    X = np.stack([np.ones(ny * nx), xs.ravel(), ys.ravel()], axis=1)

    out_median = np.empty_like(stack)
    out_plane = np.empty_like(stack)
    for z in range(nz):
        sl = stack[z]
        sl0 = sl - np.nanmedian(sl)
        out_median[z] = sl0
        beta, *_ = np.linalg.lstsq(X, sl0.ravel(), rcond=None)
        plane = (X @ beta).reshape(ny, nx)
        out_plane[z] = sl0 - plane
    return out_median, out_plane


def describe_stack(stack: np.ndarray) -> dict:
    stack = np.asarray(stack)
    payload: dict = {
        "dtype": str(stack.dtype),
        "iscomplex": bool(np.iscomplexobj(stack)),
        "shape": list(stack.shape),
        "min": float(np.nanmin(stack)),
        "max": float(np.nanmax(stack)),
        "mean": float(np.nanmean(stack)),
        "std": float(np.nanstd(stack)),
        "per_slice": [],
    }
    for z in range(stack.shape[0]):
        sl = stack[z]
        payload["per_slice"].append(
            {
                "z": int(z),
                "min": float(np.nanmin(sl)),
                "max": float(np.nanmax(sl)),
                "mean": float(np.nanmean(sl)),
                "std": float(np.nanstd(sl)),
            }
        )
    return payload


def main() -> int:
    run_dir = Path("runs/pyrex_atomic_phase_test")
    obj_path = run_dir / "stages" / "dip_free_extra" / "obj_cropped.npy"
    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = np.load(obj_path)
    stats = describe_stack(obj)
    (out_dir / "object_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    projected = np.asarray(obj, dtype=np.float64).sum(axis=0)
    vmin, vmax = robust_clim(obj, 0.5, 99.5)
    pvmin, pvmax = robust_clim(projected, 0.5, 99.5)
    titles = [f"slice {z:02d}" for z in range(obj.shape[0])]
    save_montage(
        obj,
        out_dir / "RAW_all_slices.png",
        titles=titles,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cols=4,
        suptitle="RAW potential/phase per slice (stored object)",
    )
    save_single(
        projected,
        out_dir / "RAW_projected_potential.png",
        title="RAW projected potential/phase (sum over slices)",
        cmap="viridis",
        vmin=pvmin,
        vmax=pvmax,
        dpi=300,
    )

    _median_sub, plane_removed = detrend_stack(obj)
    dproj = plane_removed.sum(axis=0)
    dvmin, dvmax = robust_clim(plane_removed, 0.5, 99.5)
    dpvmin, dpvmax = robust_clim(dproj, 0.5, 99.5)
    save_montage(
        plane_removed,
        out_dir / "DETRENDED_all_slices.png",
        titles=titles,
        cmap="viridis",
        vmin=dvmin,
        vmax=dvmax,
        cols=4,
        suptitle="DETRENDED slices (median removed + plane removed)",
    )
    save_single(
        dproj,
        out_dir / "DETRENDED_projected.png",
        title="DETRENDED projected (sum over detrended slices)",
        cmap="viridis",
        vmin=dpvmin,
        vmax=dpvmax,
        dpi=300,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
