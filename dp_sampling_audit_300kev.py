from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BFEdgeResult:
    name: str
    r_px: float | None


def robust_clim(data: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    lo, hi = np.nanpercentile(np.asarray(data, dtype=np.float64), [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


def compute_mean_dp_streaming(
    h5_path: Path,
    group: str,
    dataset: str,
    block: int = 8,
) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as f:
        ds = f[f"{group}/{dataset}"]
        ry, rx, qy, qx = ds.shape
        acc = np.zeros((qy, qx), dtype=np.float64)
        n = 0
        for y0 in range(0, ry, block):
            y1 = min(ry, y0 + block)
            for x0 in range(0, rx, block):
                x1 = min(rx, x0 + block)
                blk = np.asarray(ds[y0:y1, x0:x1, :, :], dtype=np.float64)
                acc += blk.sum(axis=(0, 1))
                n += blk.shape[0] * blk.shape[1]
        acc /= float(n)
        return acc


def center_of_mass(img: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    data = np.asarray(img, dtype=np.float64)
    if mask is not None:
        data = np.where(mask, data, 0.0)
    total = float(data.sum())
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    if total <= 0:
        return float(nx / 2.0 - 0.5), float(ny / 2.0 - 0.5)
    cy = float((data * yy).sum() / total)
    cx = float((data * xx).sum() / total)
    return cx, cy


def fit_circle_kasa(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float] | None:
    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    if x.size < 20:
        return None
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = coef
    cx = -a / 2.0
    cy = -bb / 2.0
    r = float(np.sqrt(max(0.0, cx**2 + cy**2 - c)))
    return float(cx), float(cy), r


def bf_edge_points(img: np.ndarray, center: tuple[float, float], r_guess: float) -> tuple[np.ndarray, np.ndarray]:
    from skimage.morphology import binary_erosion, disk

    data = np.asarray(img, dtype=np.float64)
    ny, nx = data.shape
    cx, cy = center
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inner = rr <= 0.70 * r_guess
    outer = rr >= 1.25 * r_guess
    vin = float(np.median(data[inner])) if inner.any() else float(np.median(data))
    vout = float(np.median(data[outer])) if outer.any() else float(np.median(data))
    thr = 0.5 * (vin + vout)
    disc = (data > thr) & (rr <= 1.6 * r_guess)
    er = binary_erosion(disc, footprint=disk(1))
    edge = disc & (~er)
    ys, xs = np.nonzero(edge)
    return xs.astype(np.float64), ys.astype(np.float64)


def radial_profile(img: np.ndarray, center: tuple[float, float], r_max: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(img, dtype=np.float64)
    ny, nx = data.shape
    cx, cy = center
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = np.floor(rr).astype(np.int32)
    if r_max is None:
        r_max = int(r_int.max())
    mask = r_int <= r_max
    r_int = r_int[mask]
    vals = data[mask]
    num = np.bincount(r_int, weights=vals, minlength=r_max + 1)
    den = np.bincount(r_int, minlength=r_max + 1)
    prof = np.divide(num, np.maximum(den, 1), dtype=np.float64)
    r = np.arange(r_max + 1, dtype=np.float64)
    return r, prof


def smooth_1d(x: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(x, dtype=np.float64)
    rad = int(np.ceil(4.0 * sigma))
    grid = np.arange(-rad, rad + 1, dtype=np.float64)
    ker = np.exp(-(grid**2) / (2.0 * sigma**2))
    ker /= float(ker.sum())
    y = np.convolve(np.asarray(x, dtype=np.float64), ker, mode="same")
    return y


def bf_radius_candidates(r: np.ndarray, prof: np.ndarray) -> list[BFEdgeResult]:
    p = smooth_1d(prof, sigma=2.0)
    g1 = np.gradient(p)
    g2 = np.gradient(g1)
    idx1 = int(np.argmin(g1[2:])) + 2 if p.size > 5 else None
    idx2 = int(np.argmax(np.abs(g2[2:-2]))) + 2 if p.size > 8 else None

    r1 = float(r[idx1]) if idx1 is not None else None
    r2 = float(r[idx2]) if idx2 is not None else None

    r0 = r1 if r1 is not None else float(r[np.argmax(p)])
    inner = p[r <= 0.6 * r0]
    outer = p[r >= 1.4 * r0]
    if inner.size < 5 or outer.size < 5:
        r3 = None
    else:
        thr = 0.5 * (float(np.median(inner)) + float(np.median(outer)))
        cross = np.where(p <= thr)[0]
        cross = cross[cross > 0]
        r3 = float(r[cross[0]]) if cross.size else None

    return [
        BFEdgeResult("max_negative_gradient", r1),
        BFEdgeResult("second_derivative_peak", r2),
        BFEdgeResult("threshold_crossing", r3),
    ]


def summarize_radius(methods: list[BFEdgeResult], r_fit: float | None) -> tuple[float | None, float | None]:
    vals = [m.r_px for m in methods if m.r_px is not None and np.isfinite(m.r_px)]
    if r_fit is not None and np.isfinite(r_fit):
        vals.append(float(r_fit))
    if len(vals) < 2:
        return (vals[0] if vals else None), None
    med = float(np.median(vals))
    mad = float(np.median(np.abs(np.asarray(vals) - med)))
    sigma = 1.4826 * mad
    if sigma <= 1e-6:
        sigma = float(np.std(vals))
    return med, sigma


def read_metadata_dp_sampling(h5_path: Path) -> float | None:
    import h5py

    keys = [
        "dp_sampling_mrad",
        "dp_sampling_mrad_per_px",
        "dp_resolution_mrad",
        "dp_resolution",
        "dpResolution",
        "dp_pixel_size_mrad",
        "dp_pixel_size",
    ]

    def probe(obj) -> float | None:
        for k in keys:
            if k in obj.attrs:
                try:
                    v = float(obj.attrs[k])
                except Exception:
                    continue
                if np.isfinite(v) and v > 0:
                    return v
        return None

    with h5py.File(h5_path, "r") as f:
        for path in ("data", "data/frames"):
            if path in f:
                v = probe(f[path])
                if v is not None:
                    return v
        v = probe(f)
        return v


def main() -> int:
    h5_path = Path(
        "/public/home/wangkai/shiyunZhang/pyrex/PTYREX_8_FOV5.0nm_def0nm_alpha25.0mrad_dx0.020nm_R256x256_Q256x256_CL0.00082m_ro73deg_trueCL87.h5"
    )
    out_dir = Path("runs/dp_sampling_audit_300kev")
    out_dir.mkdir(parents=True, exist_ok=True)

    group = "data"
    dataset = "frames"
    alpha_mrad = 25.0
    beam_energy_keV = 300.0

    dp = compute_mean_dp_streaming(h5_path, group=group, dataset=dataset, block=8)
    np.save(out_dir / "00_mean_dp.npy", dp.astype(np.float32))

    ny, nx = dp.shape
    center_geom = (nx / 2.0 - 0.5, ny / 2.0 - 0.5)
    max_idx = np.unravel_index(int(np.argmax(dp)), dp.shape)
    max_pos = (float(max_idx[1]), float(max_idx[0]))

    lo, hi = robust_clim(dp, 0.5, 99.5)
    mask_com = dp > np.percentile(dp, 75.0)
    cx_com, cy_com = center_of_mass(dp, mask=mask_com)

    r0 = 21.0
    xs, ys = bf_edge_points(dp, center=(cx_com, cy_com), r_guess=r0)
    fit = fit_circle_kasa(xs, ys)
    if fit is None:
        cx_fit, cy_fit, r_fit = cx_com, cy_com, None
        edge_residual_px = None
    else:
        cx_fit, cy_fit, r_fit = fit
        rr = np.sqrt((xs - cx_fit) ** 2 + (ys - cy_fit) ** 2)
        edge_residual_px = float(np.std(rr - float(r_fit))) if r_fit else None

    r, prof = radial_profile(dp, center=(cx_fit, cy_fit), r_max=128)
    cand = bf_radius_candidates(r, prof)
    r_est, r_unc = summarize_radius(cand, r_fit)

    dp_sampling_cal = (alpha_mrad / r_est) if r_est else None
    dp_sampling_meta = read_metadata_dp_sampling(h5_path)
    if dp_sampling_cal and dp_sampling_meta:
        rel = 100.0 * abs(dp_sampling_cal - dp_sampling_meta) / dp_sampling_meta
    else:
        rel = None

    if r_est:
        boundary = float(min(cx_fit, (nx - 1) - cx_fit, cy_fit, (ny - 1) - cy_fit))
        bf_disc_clipped = bool(boundary < 1.05 * float(r_est))
    else:
        bf_disc_clipped = None

    center_window = dp[
        int(max(0, np.floor(cy_fit) - 2)) : int(min(ny, np.floor(cy_fit) + 3)),
        int(max(0, np.floor(cx_fit) - 2)) : int(min(nx, np.floor(cx_fit) + 3)),
    ]
    p999 = float(np.percentile(dp, 99.9))
    center_saturated = bool(np.any(center_window >= p999)) if center_window.size else None

    dp_crops = [160, 192, 224, 256]
    theta_max = {}
    for n in dp_crops:
        if dp_sampling_cal:
            theta_max[str(n)] = float((n / 2.0) * dp_sampling_cal)
        else:
            theta_max[str(n)] = None

    if dp_sampling_cal:
        status = "PASS" if rel is not None and rel < 2.0 else "WARN" if rel is not None and rel < 10.0 else "FAIL"
    else:
        status = "FAIL"

    recommended_dp_crop = 192
    if dp_sampling_cal:
        for n in (192, 224, 256):
            th = float((n / 2.0) * dp_sampling_cal)
            if th / alpha_mrad >= 2.5:
                recommended_dp_crop = n
                break

    payload = {
        "beam_energy_keV": beam_energy_keV,
        "alpha_mrad": alpha_mrad,
        "dp_shape_raw": [int(ny), int(nx)],
        "mean_dp_stats": {
            "dtype": str(dp.dtype),
            "min": float(np.min(dp)),
            "max": float(np.max(dp)),
            "mean": float(np.mean(dp)),
            "std": float(np.std(dp)),
            "robust_clim_0p5_99p5": [lo, hi],
        },
        "centers_px": {
            "geom": [float(center_geom[0]), float(center_geom[1])],
            "com": [float(cx_com), float(cy_com)],
            "max": [float(max_pos[0]), float(max_pos[1])],
            "fit": [float(cx_fit), float(cy_fit)],
        },
        "bf_disc_checks": {
            "bf_disc_clipped": bf_disc_clipped,
            "center_saturated_flag": center_saturated,
            "edge_fit_residual_px": edge_residual_px,
        },
        "bf_radius_methods_px": {m.name: (None if m.r_px is None else float(m.r_px)) for m in cand},
        "bf_radius_fit_px": (None if r_fit is None else float(r_fit)),
        "bf_radius_px": (None if r_est is None else float(r_est)),
        "bf_radius_uncertainty_px": (None if r_unc is None else float(r_unc)),
        "dp_sampling_calibrated_mrad_per_px": (None if dp_sampling_cal is None else float(dp_sampling_cal)),
        "dp_sampling_metadata_mrad_per_px": (None if dp_sampling_meta is None else float(dp_sampling_meta)),
        "relative_difference_percent": (None if rel is None else float(rel)),
        "dp_crop_tested": dp_crops,
        "theta_max_mrad": theta_max,
        "recommended_dp_crop": recommended_dp_crop,
        "recommended_dp_sampling_mrad_per_px": (None if dp_sampling_cal is None else float(dp_sampling_cal)),
        "calibration_status": status,
    }
    (out_dir / "dp_sampling_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
    ax.imshow(dp, cmap="inferno", origin="lower", vmin=lo, vmax=hi)
    ax.set_title("mean DP (raw), robust 0.5–99.5%", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "00_mean_dp_raw.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
    ax.imshow(np.log1p(dp), cmap="inferno", origin="lower")
    ax.set_title("mean DP log(1+I)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "00_mean_dp_log.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 6.0), dpi=300)
    ax.imshow(np.log1p(dp), cmap="inferno", origin="lower")
    ax.plot(center_geom[0], center_geom[1], marker="+", ms=14, mew=2, color="cyan", label="geom")
    ax.plot(cx_com, cy_com, marker="x", ms=10, mew=2, color="lime", label="CoM")
    ax.plot(cx_fit, cy_fit, marker="o", ms=7, mew=2, mfc="none", color="white", label="circle fit")
    ax.set_title("BF center detection on log mean DP", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "01_bf_center_detection.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    ax.plot(r, prof, lw=1.2, label="I(r)")
    ps = smooth_1d(prof, sigma=2.0)
    ax.plot(r, ps, lw=1.2, label="smoothed")
    ax.set_xlabel("r (px)")
    ax.set_ylabel("mean intensity")
    ax.set_title("Radial profile around fitted BF center", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "02_radial_profile.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 6.0), dpi=300)
    ax.imshow(np.log1p(dp), cmap="inferno", origin="lower")
    if r_est:
        circ = plt.Circle((cx_fit, cy_fit), r_est, fill=False, ec="white", lw=1.5, alpha=0.9)
        ax.add_patch(circ)
    ax.set_title("BF edge fit overlay (log mean DP)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "03_bf_edge_fit.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 6.0), dpi=300)
    ax.imshow(np.log1p(dp), cmap="inferno", origin="lower")
    if r_est:
        ax.add_patch(plt.Circle((cx_fit, cy_fit), r_est, fill=False, ec="white", lw=1.2, alpha=0.9))
    sizes = [160, 192, 224, 256]
    colors = ["cyan", "lime", "orange", "magenta"]
    for n, c in zip(sizes, colors):
        half = n / 2.0
        x0, x1 = cx_fit - half, cx_fit + half
        y0, y1 = cy_fit - half, cy_fit + half
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=c, lw=1.1, alpha=0.85)
    ax.set_title("DP crop overlay candidates + BF circle", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "04_dp_crop_192_overlay.png", bbox_inches="tight")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
