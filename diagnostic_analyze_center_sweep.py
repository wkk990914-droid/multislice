from __future__ import annotations

import csv
import json
import re
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
        return {"peak_to_background": None, "dominant_r_px": None, "dominant_angle_deg": None}
    bg = float(np.median(p[ring]))
    pk = float(np.max(p[ring]))
    idx = np.argmax(np.where(ring, p, -np.inf))
    iy, ix = np.unravel_index(idx, p.shape)
    ang = float(np.degrees(np.arctan2(iy - cy, ix - cx)) % 360)
    dom_r = float(r[iy, ix])
    return {
        "peak_to_background": float(pk / bg) if bg > 0 else None,
        "dominant_r_px": dom_r,
        "dominant_angle_deg": ang,
    }


def tile_metrics(image: np.ndarray, tiles: int = 4) -> dict:
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    ty = ny // tiles
    tx = nx // tiles
    peak_bg = np.zeros((tiles, tiles), dtype=np.float64)
    dom_r = np.zeros((tiles, tiles), dtype=np.float64)
    dom_ang = np.zeros((tiles, tiles), dtype=np.float64)

    for iy in range(tiles):
        for ix in range(tiles):
            tile = img[iy * ty : (iy + 1) * ty, ix * tx : (ix + 1) * tx]
            m = fft_peak_metrics(fft_power(tile))
            peak_bg[iy, ix] = m["peak_to_background"] or 0.0
            dom_r[iy, ix] = m["dominant_r_px"] or 0.0
            dom_ang[iy, ix] = m["dominant_angle_deg"] or 0.0

    return {
        "tiles": tiles,
        "peak_to_bg": peak_bg,
        "dominant_r_px": dom_r,
        "dominant_angle_deg": dom_ang,
        "median_peak_to_bg": float(np.median(peak_bg)),
        "min_peak_to_bg": float(np.min(peak_bg)),
        "max_peak_to_bg": float(np.max(peak_bg)),
        "dominant_r_px_std": float(np.std(dom_r)),
    }


def lattice_coverage(tile_peak_to_bg: np.ndarray, threshold: float = 10.0) -> float:
    p = np.asarray(tile_peak_to_bg, dtype=np.float64)
    ok = p > float(threshold)
    return float(ok.sum() / ok.size)


def low_freq_fraction(power: np.ndarray, frac: float = 0.12) -> float:
    p = np.asarray(power, dtype=np.float64)
    ny, nx = p.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core = r < frac * min(ny, nx)
    tot = float(p.sum())
    return float(p[core].sum() / tot) if tot > 0 else 1.0


def edge_artifact_score(image: np.ndarray, margin_frac: float = 0.12) -> float:
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    my = int(np.round(margin_frac * ny))
    mx = int(np.round(margin_frac * nx))
    if my < 1 or mx < 1:
        return 0.0
    edge = np.zeros_like(img, dtype=bool)
    edge[:my, :] = True
    edge[-my:, :] = True
    edge[:, :mx] = True
    edge[:, -mx:] = True
    core = ~edge
    se = float(np.std(img[edge])) if edge.any() else 0.0
    sc = float(np.std(img[core])) if core.any() else 0.0
    if sc <= 0:
        return float("inf") if se > 0 else 0.0
    return float(se / sc)


def d_spacing_from_r(r_px: float, n: int, sampling_a: float) -> float | None:
    if r_px <= 0:
        return None
    dq = 1.0 / (n * sampling_a)
    freq = r_px * dq
    if freq <= 0:
        return None
    return float(1.0 / freq)


def parse_tag(name: str) -> tuple[int, int]:
    m = re.match(r"dx([+-]?\d+)_dy([+-]?\d+)", name)
    if not m:
        raise ValueError(f"bad tag: {name}")
    return int(m.group(1)), int(m.group(2))


@dataclass
class Record:
    dx: int
    dy: int
    center_cx: float
    center_cy: float
    dp_crop: tuple[int, int, int, int]
    iters: int
    initial_loss: float | None
    final_loss: float | None
    loss_reduction_pct: float | None
    fft_peak_to_bg: float | None
    uniformity_median: float
    uniformity_min: float
    lattice_coverage: float
    low_freq_frac: float
    edge_score: float
    dominant_d_A: float | None
    dominant_d_std_A: float | None


def plot_heatmap(grid: np.ndarray, xs: list[int], ys: list[int], path: Path, title: str, cmap: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=260)
    im = ax.imshow(grid, origin="lower", cmap=cmap)
    ax.set_xticks(range(len(xs)))
    ax.set_yticks(range(len(ys)))
    ax.set_xticklabels(xs)
    ax.set_yticklabels(ys)
    ax.set_xlabel("dx (px)")
    ax.set_ylabel("dy (px)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def montage(images: dict[tuple[int, int], np.ndarray], xs: list[int], ys: list[int], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin, vmax = robust_clim(np.stack([images[(dx, dy)] for dy in ys for dx in xs]))
    fig, axes = plt.subplots(len(ys), len(xs), figsize=(2.0 * len(xs), 2.0 * len(ys)), dpi=250)
    for j, dy in enumerate(ys):
        for i, dx in enumerate(xs):
            ax = axes[j, i]
            ax.imshow(images[(dx, dy)], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(f"{dx:+d},{dy:+d}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    base_run = Path("runs/pyrex_atomic_phase_test")
    sweep_dir = base_run / "diagnostics" / "center_sweep" / "coarse"
    out_dir = base_run / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    dp_diag = json.loads((out_dir / "dp_center_diagnostics.json").read_text(encoding="utf-8"))
    center_com = (float(dp_diag["center_com"]["cx"]), float(dp_diag["center_com"]["cy"]))

    records: list[Record] = []
    images: dict[tuple[int, int], np.ndarray] = {}
    xs = [-2, -1, 0, 1, 2]
    ys = [-2, -1, 0, 1, 2]

    for tag_dir in sorted(sweep_dir.glob("dx*_dy*")):
        dx, dy = parse_tag(tag_dir.name)
        metrics_path = tag_dir / "stages" / "single_slice" / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        loss = np.load(tag_dir / "stages" / "single_slice" / "loss_history.npy").astype(np.float64)
        obj = np.load(tag_dir / "stages" / "single_slice" / "obj_cropped.npy")
        img = np.asarray(obj[0], dtype=np.float64)
        images[(dx, dy)] = img

        init = m.get("initial_loss")
        fin = m.get("final_loss")
        red = None
        if init is not None and fin is not None and float(init) != 0:
            red = 100.0 * (float(init) - float(fin)) / float(init)

        sampling_a = float(m.get("object_sampling_A")) if m.get("object_sampling_A") else None
        if not sampling_a:
            ref = json.loads(
                (base_run / "stages" / "dip_free_extra" / "metrics.json").read_text(encoding="utf-8")
            )
            sampling_a = float(ref.get("object_sampling_A", 0.0)) or None
        power = fft_power(img)
        f = fft_peak_metrics(power)
        t = tile_metrics(img, tiles=4)
        cov = lattice_coverage(t["peak_to_bg"], threshold=10.0)
        lf = low_freq_fraction(power, frac=0.12)
        edge = edge_artifact_score(img)

        dom_d = None
        dom_d_std = None
        if sampling_a and f["dominant_r_px"]:
            dom_d = d_spacing_from_r(float(f["dominant_r_px"]), img.shape[0], sampling_a)
            ds = []
            for rpx in t["dominant_r_px"].ravel():
                d = d_spacing_from_r(float(rpx), img.shape[0] // 4, sampling_a)
                if d:
                    ds.append(d)
            if ds:
                dom_d_std = float(np.std(ds))

        center = (center_com[0] + dx, center_com[1] + dy)
        qy0 = int(np.round(center[1] - 96))
        qx0 = int(np.round(center[0] - 96))
        dp_crop = (qy0, qy0 + 192, qx0, qx0 + 192)

        records.append(
            Record(
                dx=dx,
                dy=dy,
                center_cx=float(center[0]),
                center_cy=float(center[1]),
                dp_crop=dp_crop,
                iters=int(loss.size),
                initial_loss=float(init) if init is not None else None,
                final_loss=float(fin) if fin is not None else None,
                loss_reduction_pct=float(red) if red is not None else None,
                fft_peak_to_bg=f["peak_to_background"],
                uniformity_median=float(t["median_peak_to_bg"]),
                uniformity_min=float(t["min_peak_to_bg"]),
                lattice_coverage=float(cov),
                low_freq_frac=float(lf),
                edge_score=float(edge),
                dominant_d_A=dom_d,
                dominant_d_std_A=dom_d_std,
            )
        )

    csv_path = out_dir / "center_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(
            [
                "dx",
                "dy",
                "center_cx_Q256",
                "center_cy_Q256",
                "dp_crop(qy0,qy1,qx0,qx1)",
                "iters",
                "initial_loss",
                "final_loss",
                "loss_reduction_pct",
                "fft_peak_to_bg",
                "tile_uniformity_median",
                "tile_uniformity_min",
                "lattice_coverage_tiles_frac",
                "low_freq_power_frac",
                "edge_artifact_score",
                "dominant_d_A",
                "dominant_d_std_A",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.dx,
                    r.dy,
                    f"{r.center_cx:.6f}",
                    f"{r.center_cy:.6f}",
                    str(tuple(int(v) for v in r.dp_crop)),
                    r.iters,
                    f"{r.initial_loss:.10g}" if r.initial_loss is not None else "",
                    f"{r.final_loss:.10g}" if r.final_loss is not None else "",
                    f"{r.loss_reduction_pct:.6g}" if r.loss_reduction_pct is not None else "",
                    f"{r.fft_peak_to_bg:.6g}" if r.fft_peak_to_bg is not None else "",
                    f"{r.uniformity_median:.6g}",
                    f"{r.uniformity_min:.6g}",
                    f"{r.lattice_coverage:.6g}",
                    f"{r.low_freq_frac:.6g}",
                    f"{r.edge_score:.6g}",
                    f"{r.dominant_d_A:.6g}" if r.dominant_d_A is not None else "",
                    f"{r.dominant_d_std_A:.6g}" if r.dominant_d_std_A is not None else "",
                ]
            )

    def grid_of(getter):
        grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
        rec = {(r.dx, r.dy): r for r in records}
        for j, dy in enumerate(ys):
            for i, dx in enumerate(xs):
                if (dx, dy) in rec:
                    grid[j, i] = float(getter(rec[(dx, dy)]))
        return grid

    loss_grid = grid_of(lambda r: r.final_loss if r.final_loss is not None else np.nan)
    uni_grid = grid_of(lambda r: r.uniformity_median)
    cov_grid = grid_of(lambda r: r.lattice_coverage)

    plot_heatmap(loss_grid, xs, ys, out_dir / "center_sweep_heatmap_loss.png", "final loss", "viridis")
    plot_heatmap(uni_grid, xs, ys, out_dir / "center_sweep_heatmap_uniformity.png", "tile FFT uniformity (median peak/bg)", "magma")
    plot_heatmap(cov_grid, xs, ys, out_dir / "center_sweep_heatmap_coverage.png", "lattice coverage (tile fraction)", "magma")

    if len(images) == 25:
        montage(images, xs, ys, out_dir / "center_sweep_montage.png")

    ranked = sorted(
        records,
        key=lambda r: (
            -(r.lattice_coverage),
            -(r.uniformity_median),
            -(r.uniformity_min),
            -(r.fft_peak_to_bg or 0.0),
            (r.final_loss or float("inf")),
            (r.low_freq_frac),
            (r.edge_score),
        ),
    )
    summary = {
        "center_com": {"cx": center_com[0], "cy": center_com[1]},
        "best": ranked[0].__dict__ if ranked else None,
        "top5": [r.__dict__ for r in ranked[:5]],
        "n_done": len(records),
    }
    (out_dir / "center_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
