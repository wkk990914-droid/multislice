from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _to_numpy(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _median_step(values: np.ndarray) -> float | None:
    uniq = np.unique(np.asarray(values, dtype=np.float64))
    if uniq.size < 2:
        return None
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def audit_scan_positions(out_dir: Path) -> dict:
    from ptycho.config import PipelineConfig
    from ptycho.worker import prepare_dataset

    cfg = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    _dset, pdset = prepare_dataset(cfg)

    pos_px = _to_numpy(pdset.scan_positions_px)
    scan_sampling_cfg = float(cfg.experiment.scan_step_size)
    units = ("A", "A")

    metrics_path = Path("runs/pyrex_atomic_phase_test/stages/dip_free_extra/metrics.json")
    object_sampling = None
    if metrics_path.exists():
        try:
            object_sampling = float(json.loads(metrics_path.read_text(encoding="utf-8")).get("object_sampling_A"))
        except Exception:
            object_sampling = None
    if not object_sampling or object_sampling <= 0:
        object_sampling = scan_sampling_cfg

    if cfg.data.scan_crop:
        y0, y1, x0, x1 = (int(v) for v in cfg.data.scan_crop)
        scan_ny = y1 - y0
        scan_nx = x1 - x0
    else:
        scan_ny, scan_nx = (int(cfg.data.scan_shape[0]), int(cfg.data.scan_shape[1]))

    pos_A = pos_px * float(object_sampling)
    xA = pos_A[:, 1]
    yA = pos_A[:, 0]

    median_dx_px = None
    median_dy_px = None
    median_dx_A = None
    median_dy_A = None
    if pos_A.shape[0] == scan_ny * scan_nx:
        grid_px = pos_px.reshape(scan_ny, scan_nx, 2)
        dx = grid_px[:, 1:, :] - grid_px[:, :-1, :]
        dy = grid_px[1:, :, :] - grid_px[:-1, :, :]
        dx_norm_px = np.linalg.norm(dx, axis=-1)
        dy_norm_px = np.linalg.norm(dy, axis=-1)
        median_dx_px = float(np.median(dx_norm_px))
        median_dy_px = float(np.median(dy_norm_px))
        median_dx_A = float(median_dx_px * object_sampling)
        median_dy_A = float(median_dy_px * object_sampling)

    payload = {
        "n_positions": int(pos_px.shape[0]),
        "scan_positions_px_shape": list(pos_px.shape),
        "scan_grid_shape": [int(scan_ny), int(scan_nx)],
        "scan_step_cfg_A": scan_sampling_cfg,
        "object_sampling_A": float(object_sampling),
        "scan_units": list(units),
        "x_A_min": float(np.min(xA)),
        "x_A_max": float(np.max(xA)),
        "y_A_min": float(np.min(yA)),
        "y_A_max": float(np.max(yA)),
        "median_dx_px": median_dx_px,
        "median_dy_px": median_dy_px,
        "median_dx_A": median_dx_A,
        "median_dy_A": median_dy_A,
        "n_unique_x": int(np.unique(xA).size),
        "n_unique_y": int(np.unique(yA).size),
        "fov_x_A": float(np.max(xA) - np.min(xA)),
        "fov_y_A": float(np.max(yA) - np.min(yA)),
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=250)
    ax.scatter(xA, yA, s=2.0, c=np.arange(xA.size), cmap="viridis", linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x [{units[1]}]")
    ax.set_ylabel(f"y [{units[0]}]")
    ax.set_title("scan positions after cropping + preprocess (Quantem pdset.scan_positions_px)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "scan_positions.png", bbox_inches="tight")
    plt.close(fig)

    return payload


def compute_mean_dp_h5(
    path: Path,
    group: str,
    dataset: str,
    scan_crop: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    import h5py

    y0, y1, x0, x1 = scan_crop
    with h5py.File(path, "r") as f:
        ds = f[f"{group}/{dataset}"]
        shape = tuple(int(s) for s in ds.shape)
        acc = np.zeros((shape[2], shape[3]), dtype=np.float64)
        n = 0
        for yy in range(y0, y1):
            block = ds[yy, x0:x1, :, :]
            acc += np.asarray(block, dtype=np.float64).sum(axis=0)
            n += block.shape[0]
    acc /= float(n)
    return shape, acc


def estimate_bf_center(mean_dp: np.ndarray, guess: tuple[float, float], radius_px: float) -> tuple[float, float]:
    ny, nx = mean_dp.shape
    gy, gx = guess
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.sqrt((yy - gy) ** 2 + (xx - gx) ** 2)
    mask = rr <= float(radius_px)
    w = np.asarray(mean_dp, dtype=np.float64)
    w = np.where(mask, w, 0.0)
    total = float(w.sum())
    if total <= 0:
        return float(gy), float(gx)
    cy = float((w * yy).sum() / total)
    cx = float((w * xx).sum() / total)
    return cy, cx


def audit_diffraction(out_dir: Path) -> dict:
    from ptycho.config import PipelineConfig

    cfg = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
    path = cfg.data.resolved_path()
    group = cfg.data.group or "data"
    dataset = cfg.data.dataset or "frames"
    scan_crop = tuple(int(v) for v in cfg.data.scan_crop) if cfg.data.scan_crop else (0, 256, 0, 256)
    dp_crop = tuple(int(v) for v in cfg.data.dp_crop) if cfg.data.dp_crop else None

    shape, mean_dp = compute_mean_dp_h5(path, group, dataset, scan_crop=scan_crop)
    ny, nx = mean_dp.shape
    guess = (ny / 2 - 0.5, nx / 2 - 0.5)
    bf_center = estimate_bf_center(mean_dp, guess=guess, radius_px=35.0)

    dp_sampling = float(cfg.data.dp_sampling_mrad) if cfg.data.dp_sampling_mrad else None
    alpha = float(cfg.experiment.probe_semiangle)
    bf_r = alpha / dp_sampling if (dp_sampling and dp_sampling > 0) else None

    payload = {
        "h5_path": str(path),
        "dataset": f"{group}/{dataset}",
        "dataset_shape": list(shape),
        "mean_dp_shape": [int(ny), int(nx)],
        "geometric_center_px": [float(guess[0]), float(guess[1])],
        "estimated_bf_center_px": [float(bf_center[0]), float(bf_center[1])],
        "dp_sampling_mrad_per_px": dp_sampling,
        "alpha_mrad": alpha,
        "expected_bf_radius_px": bf_r,
        "dp_crop": list(dp_crop) if dp_crop else None,
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.log1p(mean_dp.astype(np.float64))
    fig, ax = plt.subplots(figsize=(6.6, 6.2), dpi=260)
    im = ax.imshow(img, cmap="inferno", origin="lower")
    ax.plot(guess[1], guess[0], marker="+", ms=14, mew=2, color="cyan", label="geometric center")
    ax.plot(bf_center[1], bf_center[0], marker="x", ms=10, mew=2, color="lime", label="BF CoM")
    if bf_r:
        circ = plt.Circle((guess[1], guess[0]), bf_r, fill=False, ec="cyan", lw=1.5, alpha=0.9)
        ax.add_patch(circ)
    if dp_crop:
        qy0, qy1, qx0, qx1 = dp_crop
        ax.plot([qx0, qx1, qx1, qx0, qx0], [qy0, qy0, qy1, qy1, qy0], color="white", lw=1.2)
    ax.set_title("mean diffraction (log1p) with chosen center / BF radius / 192 crop")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.65)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_dp_with_center.png", bbox_inches="tight")
    plt.close(fig)

    return payload


def main() -> int:
    out_dir = Path("runs/pyrex_atomic_phase_test/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    scan = audit_scan_positions(out_dir)
    diff = audit_diffraction(out_dir)
    (out_dir / "scan_audit.json").write_text(json.dumps(scan, indent=2), encoding="utf-8")
    (out_dir / "diffraction_audit.json").write_text(json.dumps(diff, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
