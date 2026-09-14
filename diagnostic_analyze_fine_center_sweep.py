from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from diagnostic_analyze_center_sweep import (
    edge_artifact_score,
    fft_peak_metrics,
    fft_power,
    lattice_coverage,
    low_freq_fraction,
    robust_clim,
    tile_metrics,
)


def parse_tag(name: str) -> tuple[int, int]:
    m = re.match(r"dx([+-]?\d+)_dy([+-]?\d+)", name)
    if not m:
        raise ValueError(f"bad tag: {name}")
    return int(m.group(1)), int(m.group(2))


def crop_for_center(center: tuple[float, float], size: int = 192) -> tuple[int, int, int, int]:
    cx, cy = center
    half = size / 2.0
    qx0 = int(np.round(cx - half))
    qy0 = int(np.round(cy - half))
    return (qy0, qy0 + size, qx0, qx0 + size)


def plot_heatmap(grid: np.ndarray, xs: list[int], ys: list[int], path: Path, title: str, cmap: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=260)
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
    out_dir = base_run / "diagnostics"
    fine_dir = out_dir / "center_sweep" / "fine"

    dp_diag = json.loads((out_dir / "dp_center_diagnostics.json").read_text(encoding="utf-8"))
    center_com = (float(dp_diag["center_com"]["cx"]), float(dp_diag["center_com"]["cy"]))

    summary = json.loads((out_dir / "center_sweep_summary.json").read_text(encoding="utf-8"))
    coarse_best = summary["best"]
    coarse_dx = int(coarse_best["dx"])
    coarse_dy = int(coarse_best["dy"])

    records = []
    images: dict[tuple[int, int], np.ndarray] = {}
    for tag_dir in sorted(fine_dir.glob("dx*_dy*")):
        dx, dy = parse_tag(tag_dir.name)
        metrics_path = tag_dir / "stages" / "single_slice" / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        loss = np.load(tag_dir / "stages" / "single_slice" / "loss_history.npy").astype(np.float64)
        obj = np.load(tag_dir / "stages" / "single_slice" / "obj_cropped.npy")
        img = np.asarray(obj[0], dtype=np.float64)
        images[(dx, dy)] = img

        init = float(m["initial_loss"])
        fin = float(m["final_loss"])
        red = 100.0 * (init - fin) / init if init != 0 else None

        power = fft_power(img)
        f = fft_peak_metrics(power)
        t = tile_metrics(img, tiles=4)
        cov = lattice_coverage(t["peak_to_bg"], threshold=10.0)
        lf = low_freq_fraction(power, frac=0.12)
        edge = edge_artifact_score(img)

        records.append(
            {
                "dx": dx,
                "dy": dy,
                "center_cx": center_com[0] + dx,
                "center_cy": center_com[1] + dy,
                "dp_crop": crop_for_center((center_com[0] + dx, center_com[1] + dy), size=192),
                "iters": int(loss.size),
                "initial_loss": init,
                "final_loss": fin,
                "loss_reduction_pct": red,
                "fft_peak_to_bg": f["peak_to_background"],
                "uniformity_median": t["median_peak_to_bg"],
                "uniformity_min": t["min_peak_to_bg"],
                "lattice_coverage": cov,
                "low_freq_frac": lf,
                "edge_score": edge,
            }
        )

    ranked = sorted(
        records,
        key=lambda r: (
            -(r["lattice_coverage"]),
            -(r["uniformity_median"]),
            -(r["uniformity_min"]),
            -(r["fft_peak_to_bg"] or 0.0),
            (r["final_loss"]),
            (r["low_freq_frac"]),
            (r["edge_score"]),
        ),
    )
    best = ranked[0] if ranked else None
    (out_dir / "fine_center_best.json").write_text(json.dumps({"best": best, "all": records}, indent=2), encoding="utf-8")

    csv_path = out_dir / "fine_center_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(
            [
                "dx",
                "dy",
                "center_cx_Q256",
                "center_cy_Q256",
                "dp_crop(qy0,qy1,qx0,qx1)",
                "final_loss",
                "loss_reduction_pct",
                "fft_peak_to_bg",
                "tile_uniformity_median",
                "tile_uniformity_min",
                "lattice_coverage_tiles_frac",
                "low_freq_power_frac",
                "edge_artifact_score",
            ]
        )
        for r in ranked:
            w.writerow(
                [
                    r["dx"],
                    r["dy"],
                    f"{r['center_cx']:.6f}",
                    f"{r['center_cy']:.6f}",
                    str(tuple(int(v) for v in r["dp_crop"])),
                    f"{r['final_loss']:.10g}",
                    f"{r['loss_reduction_pct']:.6g}" if r["loss_reduction_pct"] is not None else "",
                    f"{r['fft_peak_to_bg']:.6g}" if r["fft_peak_to_bg"] is not None else "",
                    f"{r['uniformity_median']:.6g}",
                    f"{r['uniformity_min']:.6g}",
                    f"{r['lattice_coverage']:.6g}",
                    f"{r['low_freq_frac']:.6g}",
                    f"{r['edge_score']:.6g}",
                ]
            )

    xs = sorted({r["dx"] for r in records})
    ys = sorted({r["dy"] for r in records})
    rec = {(r["dx"], r["dy"]): r for r in records}

    def grid_of(key: str) -> np.ndarray:
        g = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
        for j, dy in enumerate(ys):
            for i, dx in enumerate(xs):
                if (dx, dy) in rec:
                    g[j, i] = float(rec[(dx, dy)][key])
        return g

    plot_heatmap(grid_of("final_loss"), xs, ys, out_dir / "fine_center_heatmap_loss.png", "fine center: final loss", "viridis")
    plot_heatmap(
        grid_of("uniformity_median"),
        xs,
        ys,
        out_dir / "fine_center_heatmap_uniformity.png",
        "fine center: tile FFT uniformity (median peak/bg)",
        "magma",
    )
    plot_heatmap(
        grid_of("lattice_coverage"),
        xs,
        ys,
        out_dir / "fine_center_heatmap_coverage.png",
        "fine center: lattice coverage (tile fraction)",
        "magma",
    )

    if len(images) == len(xs) * len(ys):
        montage(images, xs, ys, out_dir / "fine_center_montage.png")

    if best:
        best_dp = {
            "coarse_best_dxdy": {"dx": coarse_dx, "dy": coarse_dy},
            "fine_best_dxdy": {"dx": int(best["dx"]), "dy": int(best["dy"])},
            "BEST_DP_CENTER_Q256": {"cx": float(best["center_cx"]), "cy": float(best["center_cy"])},
            "BEST_DP_CENTER_offset_from_geom": {
                "dx": float(best["center_cx"] - 127.5),
                "dy": float(best["center_cy"] - 127.5),
            },
            "BEST_DP_CROP": list(int(v) for v in best["dp_crop"]),
        }
        (out_dir / "BEST_DP_CENTER.json").write_text(json.dumps(best_dp, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

