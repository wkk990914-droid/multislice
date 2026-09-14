"""Phase / amplitude analysis for multislice ptychography reconstructions.

``ObjectPixelated(obj_type="potential")`` stores a **real-valued** array which
``quantem`` turns into a wave via ``exp(1j * obj)`` (see
``ObjectPixelated._get_obj_patches``).  The stored numbers therefore *are* the
electron phase in radians, and ``np.angle`` on them would return 0 for positive
values / pi for negative ones.  :func:`decompose_object` handles both the
real-potential and the genuinely complex case so the caller never mixes them up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def to_numpy(array: Any) -> np.ndarray:
    import torch

    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


# --------------------------------------------------------------------------- #
# sanity checks
# --------------------------------------------------------------------------- #
def check_finite(name: str, *arrays: Any) -> dict[str, dict[str, Any]]:
    """Report NaN / Inf for every array; raise on the first non-finite one."""
    report: dict[str, dict[str, Any]] = {}
    for index, arr in enumerate(arrays):
        a = to_numpy(arr)
        label = name if len(arrays) == 1 else f"{name}[{index}]"
        nan = int(np.isnan(a).sum())
        inf = int(np.isinf(a).sum())
        finite = bool(np.isfinite(a).all())
        report[label] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "nan": nan,
            "inf": inf,
            "finite": finite,
            "min": float(np.nanmin(a)) if a.size else None,
            "max": float(np.nanmax(a)) if a.size else None,
        }
        if not finite:
            raise FloatingPointError(
                f"{label}: {nan} NaN and {inf} Inf values (shape {a.shape}, dtype {a.dtype})"
            )
    return report


# --------------------------------------------------------------------------- #
# object decomposition
# --------------------------------------------------------------------------- #
def decompose_object(obj: Any) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(phase, amplitude, kind)`` for a (Nz, Ny, Nx) reconstruction.

    ``kind`` is ``"complex"`` when the array is complex (phase = ``np.angle``)
    or ``"potential"`` when it is real, in which case the stored value is
    already the phase in radians and the amplitude is unity.
    """
    arr = to_numpy(obj)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D multislice object, got shape {arr.shape}")
    if np.iscomplexobj(arr):
        return np.angle(arr), np.abs(arr), "complex"
    return arr.astype(np.float64), np.ones_like(arr, dtype=np.float64), "potential"


def slice_depths(num_slices: int, slice_thickness: float) -> np.ndarray:
    """Physical depth (Angstrom) of each slice centre."""
    return (np.arange(num_slices) + 0.5) * float(slice_thickness)


# --------------------------------------------------------------------------- #
# projected phase
# --------------------------------------------------------------------------- #
def projected_phase_unwrapped(phase: np.ndarray) -> np.ndarray:
    """Method A: unwrap along the slice axis, then sum.

    ``np.unwrap`` needs a *phase* in radians along the unwrapped axis; for the
    real-potential case the stored values are already unwrapped (they can exceed
    ``pi``), so the unwrap is a no-op there and the sum is exact.
    """
    unwrapped = np.unwrap(phase, axis=0)
    return unwrapped.sum(axis=0)


def projected_phase_complex_product(
    phase: np.ndarray, amplitude: np.ndarray | None = None
) -> np.ndarray:
    """Method B: phase of the slice-wise complex transmission product."""
    unit = np.exp(1j * phase) if amplitude is None else amplitude * np.exp(1j * phase)
    prod = np.prod(unit, axis=0)
    return np.angle(prod)


def projected_amplitude(amplitude: np.ndarray) -> np.ndarray:
    return np.prod(amplitude, axis=0)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def robust_clim(data: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    lo, hi = np.nanpercentile(data, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


_robust_clim = robust_clim


def save_single(
    data: np.ndarray,
    path: Path,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 200,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


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
) -> Path:
    """Arrange ``stack`` (Nz, Ny, Nx) into a cols-wide grid with one colorbar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz = stack.shape[0]
    rows = int(np.ceil(nz / cols))
    if vmin is None or vmax is None:
        vmin, vmax = _robust_clim(stack)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.35 * cols, 2.35 * rows), squeeze=False, dpi=dpi
    )
    flat = axes.ravel()
    for z in range(nz):
        ax = flat[z]
        im = ax.imshow(stack[z], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(titles[z], fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
    for z in range(nz, rows * cols):
        flat[z].axis("off")

    for ax in axes[0, :]:
        ax.set_xticks([])
    cbar_ax = fig.add_axes([0.925, 0.12, 0.014, 0.74])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=0.985)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def save_phase_fft(
    phase_image: np.ndarray, path: Path, sampling_a: float, dpi: int = 300
) -> dict[str, Any]:
    """2D FFT of the projected phase, log-scaled, with periodic peaks flagged."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.asarray(phase_image, dtype=np.float64)
    img = img - img.mean()
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    fft = np.fft.fftshift(np.fft.fft2(img * win))
    power = np.log1p(np.abs(fft))

    ny, nx = power.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    radius_px = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Mask the central (DC / low-frequency) disc before hunting for peaks.
    core = radius_px < max(4.0, 0.03 * min(ny, nx))
    ring = (~core) & (radius_px < 0.42 * min(ny, nx))
    background = float(np.nanmedian(power[ring])) if ring.any() else float(power.min())
    peak_val = float(np.nanmax(power[ring])) if ring.any() else float(power.max())
    contrast = peak_val - background

    peak_mask = ring & (power > background + 0.55 * contrast)
    ys, xs = np.nonzero(peak_mask)
    # Spatial frequency in 1/Angstrom for the strongest peak.
    dq = 1.0 / (img.shape[0] * sampling_a)
    d_spacing = None
    if len(ys):
        rr = radius_px[ys, xs].max() * dq
        if rr > 0:
            d_spacing = 1.0 / rr

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    im = ax.imshow(power, cmap="inferno", origin="lower")
    ax.plot(cx, cy, marker="+", ms=14, mew=2, color="cyan")
    if len(ys):
        ax.scatter(xs, ys, s=14, facecolors="none", edgecolors="lime", linewidths=1.0)
    ax.set_title(
        f"projected-phase FFT (log |F|)\npeak-background contrast = {contrast:.2f} (px)",
        fontsize=10,
    )
    ax.set_xlabel(f"spatial freq, {1 / (sampling_a):.0f} $\\AA^{{-1}}$/px")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": str(path),
        "power": power,
        "background": background,
        "peak": peak_val,
        "contrast": contrast,
        "num_peaks": int(len(ys)),
        "d_spacing_A": d_spacing,
    }


def save_loss_plot(
    loss_histories: dict[str, np.ndarray], path: Path, dpi: int = 200
) -> Path:
    """Concatenated per-stage loss curves on a log-y axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    offset = 0
    bounds: list[tuple[str, int, int]] = []
    for name, losses in loss_histories.items():
        losses = np.asarray(losses, dtype=np.float64).ravel()
        if losses.size == 0:
            continue
        xs = np.arange(1, losses.size + 1) + offset
        ax.semilogy(xs, losses, marker="o", ms=3, lw=1.2, label=f"{name} ({losses.size})")
        bounds.append((name, xs[0], xs[-1]))
        offset += losses.size

    for name, x0, x1 in bounds:
        ax.axvline(x0 - 0.5, color="grey", ls=":", lw=0.8)
        ax.text(
            (x0 + x1) / 2,
            ax.get_ylim()[1] * 0.92,
            name,
            ha="center",
            va="top",
            fontsize=7,
            rotation=0,
            color="dimgrey",
        )
    ax.axvline(offset - 0.5, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("cumulative iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("reconstruction loss per stage")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def write_loss_csv(path: Path, loss_histories: dict[str, np.ndarray]) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "iteration", "loss"])
        for name, losses in loss_histories.items():
            for i, value in enumerate(np.asarray(losses).ravel(), start=1):
                writer.writerow([name, i, f"{float(value):.10g}"])
    return path


def save_summary_figure(
    phase_stack: np.ndarray,
    phase_titles: list[str],
    projected: np.ndarray,
    fft_info: dict[str, Any],
    loss_histories: dict[str, np.ndarray],
    path: Path,
    vmin: float,
    vmax: float,
    dpi: int = 300,
    cols: int = 4,
) -> Path:
    """Publication-style composite: montage | projected phase | FFT | loss."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    nz = phase_stack.shape[0]
    rows = int(np.ceil(nz / cols))

    fig = plt.figure(figsize=(3.1 * cols + 4.6, 2.45 * rows), dpi=dpi)
    outer = GridSpec(1, 2, width_ratios=[3.1 * cols, 4.6], wspace=0.09, figure=fig)

    # ---- left: the slice montage ---------------------------------------- #
    left = GridSpecFromSubplotSpec(
        rows, cols, subplot_spec=outer[0, 0], hspace=0.42, wspace=0.16
    )
    last_im = None
    for z in range(nz):
        ax = fig.add_subplot(left[z // cols, z % cols])
        last_im = ax.imshow(
            phase_stack[z], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower"
        )
        ax.set_title(phase_titles[z], fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    # ---- right: projected phase, FFT, convergence ----------------------- #
    right = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 1], hspace=0.42)

    ax_proj = fig.add_subplot(right[0, 0])
    im_proj = ax_proj.imshow(projected, cmap="twilight_shifted", origin="lower")
    ax_proj.set_title("projected phase (unwrap along z, then sum)", fontsize=9)
    ax_proj.set_xticks([])
    ax_proj.set_yticks([])
    fig.colorbar(im_proj, ax=ax_proj, fraction=0.046, pad=0.04)

    ax_fft = fig.add_subplot(right[1, 0])
    power = fft_info.get("power")
    if power is None:
        power = _recompute_fft_power(projected)
    im_fft = ax_fft.imshow(power, cmap="inferno", origin="lower")
    ax_fft.set_title(
        f"phase FFT | contrast={fft_info['contrast']:.2f} | peaks={fft_info['num_peaks']}",
        fontsize=9,
    )
    ax_fft.set_xticks([])
    ax_fft.set_yticks([])
    fig.colorbar(im_fft, ax=ax_fft, fraction=0.046, pad=0.04)

    ax_loss = fig.add_subplot(right[2, 0])
    offset = 0
    for name, losses in loss_histories.items():
        losses = np.asarray(losses).ravel()
        if losses.size == 0:
            continue
        ax_loss.semilogy(
            np.arange(1, losses.size + 1) + offset, losses, lw=1.2, label=name
        )
        offset += losses.size
    ax_loss.set_title("loss convergence", fontsize=9)
    ax_loss.set_xlabel("cumulative iteration", fontsize=8)
    ax_loss.set_ylabel("loss", fontsize=8)
    ax_loss.grid(True, which="both", alpha=0.25)
    ax_loss.legend(fontsize=6)
    ax_loss.tick_params(labelsize=7)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def _recompute_fft_power(phase_image: np.ndarray) -> np.ndarray:
    from numpy.fft import fft2, fftshift

    img = np.asarray(phase_image, dtype=np.float64)
    img = img - img.mean()
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    return np.log1p(np.abs(fftshift(fft2(img * win))))
