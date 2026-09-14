from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_mean_dp_h5(
    path: Path,
    group: str,
    dataset: str,
    scan_crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        ds = f[f"{group}/{dataset}"]
        if scan_crop is None:
            dp = ds[...].mean(axis=(0, 1)).astype(np.float64)
            return dp
        y0, y1, x0, x1 = (int(v) for v in scan_crop)
        acc = np.zeros((ds.shape[2], ds.shape[3]), dtype=np.float64)
        n = 0
        for yy in range(y0, y1):
            block = ds[yy, x0:x1, :, :]
            acc += np.asarray(block, dtype=np.float64).sum(axis=0)
            n += block.shape[0]
        acc /= float(n)
        return acc


def center_of_mass(img: np.ndarray, center_guess: tuple[float, float], radius: float) -> tuple[float, float]:
    ny, nx = img.shape
    gy, gx = center_guess
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.sqrt((yy - gy) ** 2 + (xx - gx) ** 2)
    mask = rr <= float(radius)
    w = np.where(mask, np.asarray(img, dtype=np.float64), 0.0)
    total = float(w.sum())
    if total <= 0:
        return float(gx), float(gy)
    cy = float((w * yy).sum() / total)
    cx = float((w * xx).sum() / total)
    return float(cx), float(cy)


def fit_circle_kasa(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = coef
    cx = -a / 2.0
    cy = -bb / 2.0
    r = float(np.sqrt(max(0.0, cx**2 + cy**2 - c)))
    return float(cx), float(cy), r


def edge_points_for_disc(
    img: np.ndarray, center_guess: tuple[float, float], radius_guess: float
) -> tuple[np.ndarray, np.ndarray]:
    from skimage.morphology import binary_erosion, disk

    ny, nx = img.shape
    cxg, cyg = center_guess
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.sqrt((yy - cyg) ** 2 + (xx - cxg) ** 2)

    inside = rr <= 0.8 * radius_guess
    outside = rr >= 1.25 * radius_guess
    vin = float(np.median(img[inside])) if inside.any() else float(np.median(img))
    vout = float(np.median(img[outside])) if outside.any() else float(np.median(img))
    thr = 0.5 * (vin + vout)

    disc = (img > thr) & (rr <= 1.6 * radius_guess)
    er = binary_erosion(disc, footprint=disk(1))
    edge = disc & (~er)
    ys, xs = np.nonzero(edge)
    return xs.astype(np.float64), ys.astype(np.float64)


def main() -> int:
    from ptycho.config import PipelineConfig

    cfg = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    path = cfg.data.resolved_path()
    group = cfg.data.group or "data"
    dataset = cfg.data.dataset or "frames"
    scan_crop = tuple(cfg.data.scan_crop) if cfg.data.scan_crop else None

    dp = compute_mean_dp_h5(path, group, dataset, scan_crop)
    ny, nx = dp.shape
    center_geom = (nx / 2.0 - 0.5, ny / 2.0 - 0.5)

    dp_sampling = float(cfg.data.dp_sampling_mrad) if cfg.data.dp_sampling_mrad else None
    alpha = float(cfg.experiment.probe_semiangle)
    bf_radius = alpha / dp_sampling if (dp_sampling and dp_sampling > 0) else 22.0

    center_com = center_of_mass(dp, center_guess=(center_geom[1], center_geom[0]), radius=1.2 * bf_radius)
    xs, ys = edge_points_for_disc(dp, center_guess=center_com, radius_guess=bf_radius)
    if xs.size >= 20:
        cx_fit, cy_fit, r_fit = fit_circle_kasa(xs, ys)
        center_fit = (cx_fit, cy_fit)
        bf_radius_fit = r_fit
    else:
        center_fit = center_com
        bf_radius_fit = bf_radius

    dx_com = center_com[0] - center_geom[0]
    dy_com = center_com[1] - center_geom[1]
    dx_fit = center_fit[0] - center_geom[0]
    dy_fit = center_fit[1] - center_geom[1]

    payload = {
        "center_geom": {"cx": float(center_geom[0]), "cy": float(center_geom[1])},
        "center_com": {"cx": float(center_com[0]), "cy": float(center_com[1])},
        "center_fit": {"cx": float(center_fit[0]), "cy": float(center_fit[1])},
        "offset_com_from_geom": {"dx": float(dx_com), "dy": float(dy_com)},
        "offset_fit_from_geom": {"dx": float(dx_fit), "dy": float(dy_fit)},
        "bf_radius_px_expected": float(bf_radius),
        "bf_radius_px_fit": float(bf_radius_fit),
        "dp_sampling_mrad_per_px": dp_sampling,
        "alpha_mrad": alpha,
        "scan_crop": list(scan_crop) if scan_crop else None,
    }

    out_dir = Path("runs/pyrex_atomic_phase_test/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dp_center_diagnostics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.log1p(dp.astype(np.float64))
    fig, ax = plt.subplots(figsize=(6.8, 6.2), dpi=280)
    im = ax.imshow(img, cmap="inferno", origin="lower")

    ax.plot(center_geom[0], center_geom[1], marker="+", ms=14, mew=2, color="cyan", label="geom")
    ax.plot(center_com[0], center_com[1], marker="x", ms=10, mew=2, color="lime", label="BF CoM")
    ax.plot(center_fit[0], center_fit[1], marker="o", ms=7, mew=2, mfc="none", color="white", label="disc fit")

    circ = plt.Circle((center_fit[0], center_fit[1]), bf_radius_fit, fill=False, ec="white", lw=1.3, alpha=0.9)
    ax.add_patch(circ)

    if cfg.data.dp_crop:
        qy0, qy1, qx0, qx1 = (int(v) for v in cfg.data.dp_crop)
        ax.plot([qx0, qx1, qx1, qx0, qx0], [qy0, qy0, qy1, qy1, qy0], color="cyan", lw=1.2, alpha=0.9)

    def rect_for_center(center: tuple[float, float], color: str):
        cx, cy = center
        x0 = cx - 96
        x1 = cx + 96
        y0 = cy - 96
        y1 = cy + 96
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, lw=1.0, alpha=0.8, ls="--")

    rect_for_center(center_com, "lime")
    rect_for_center(center_fit, "white")

    ax.set_title("DP center diagnostics (log1p mean DP) — Q256 coordinates", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "dp_center_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

