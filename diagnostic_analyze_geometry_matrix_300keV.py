from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
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


@dataclass
class Row:
    convention: str
    rotation_deg: float
    tag: str
    final_loss: float
    loss_reduction_pct: float
    coverage: float
    uniformity_median: float
    uniformity_min: float
    dominant_d_A: float | None
    dominant_d_std_A: float | None
    low_freq_frac: float
    edge_score: float


def parse_tag(tag: str) -> tuple[str, float]:
    m = re.match(r"(.+)__rot_([pm])(\\d+)", tag)
    if not m:
        raise ValueError(tag)
    conv = m.group(1)
    sign = 1.0 if m.group(2) == "p" else -1.0
    rot = sign * float(m.group(3))
    return conv, rot


def d_spacing_from_r(r_px: float, n: int, sampling_a: float) -> float | None:
    if r_px <= 0:
        return None
    dq = 1.0 / (n * sampling_a)
    freq = r_px * dq
    if freq <= 0:
        return None
    return float(1.0 / freq)


def heatmap(values: dict[tuple[int, int], float], path: Path, title: str, cmap: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conventions = [
        "A_identity",
        "B_transpose",
        "C_flip_x",
        "D_flip_y",
        "E_transpose_flip_x",
        "F_transpose_flip_y",
        "G_flip_xy",
        "H_transpose_flip_xy",
    ]
    rotations = [73.0, 17.0, -73.0, -17.0]
    grid = np.full((len(conventions), len(rotations)), np.nan, dtype=np.float64)
    for i, c in enumerate(conventions):
        for j, r in enumerate(rotations):
            key = (i, j)
            grid[i, j] = values.get(key, np.nan)

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=260)
    im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto")
    ax.set_yticks(range(len(conventions)))
    ax.set_yticklabels(conventions, fontsize=8)
    ax.set_xticks(range(len(rotations)))
    ax.set_xticklabels([f\"{r:+.0f}°\" for r in rotations], fontsize=8)
    ax.set_xlabel("rotation")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def montage_panels(panels: dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conventions = [
        "A_identity",
        "B_transpose",
        "C_flip_x",
        "D_flip_y",
        "E_transpose_flip_x",
        "F_transpose_flip_y",
        "G_flip_xy",
        "H_transpose_flip_xy",
    ]
    rotations = [73.0, 17.0, -73.0, -17.0]
    tags = []
    for c in conventions:
        for r in rotations:
            tag = f\"{c}__rot_{r:+.0f}\".replace("+", "p").replace("-", "m")
            tags.append(tag)
    imgs = [panels[t] for t in tags if t in panels]
    vmin, vmax = robust_clim(np.stack(imgs)) if imgs else (0.0, 1.0)

    fig, axes = plt.subplots(len(conventions), len(rotations), figsize=(3.0 * len(rotations), 2.5 * len(conventions)), dpi=260)
    for i, c in enumerate(conventions):
        for j, r in enumerate(rotations):
            tag = f\"{c}__rot_{r:+.0f}\".replace("+", "p").replace("-", "m")
            ax = axes[i, j]
            if tag in panels:
                ax.imshow(panels[tag], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
                ax.set_title(f\"{c}\\n{r:+.0f}°\", fontsize=7)
            else:
                ax.axis("off")
                continue
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    base_run = Path("runs/pyrex_300keV_geometry_validation")
    matrix_root = base_run / "geometry_matrix"
    diag_root = base_run / "diagnostics"
    diag_root.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    panels: dict[str, np.ndarray] = {}

    for run_dir in sorted(matrix_root.glob("*__rot_*")):
        metrics_path = run_dir / "stages" / "single_slice" / "metrics.json"
        if not metrics_path.exists():
            continue
        tag = run_dir.name
        conv, rot = parse_tag(tag)

        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        init = float(m["initial_loss"])
        fin = float(m["final_loss"])
        red = 100.0 * (init - fin) / init if init != 0 else 0.0

        obj = np.load(run_dir / "stages" / "single_slice" / "obj_cropped.npy")
        img = np.asarray(obj[0], dtype=np.float64)
        panels[tag] = img

        power = fft_power(img)
        f = fft_peak_metrics(power)
        t = tile_metrics(img, tiles=4)
        cov = lattice_coverage(t["peak_to_bg"], threshold=10.0)
        lf = low_freq_fraction(power, frac=0.12)
        edge = edge_artifact_score(img)

        sampling_a = float(m.get("object_sampling_A", 0.0)) or None
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

        rows.append(
            Row(
                convention=conv,
                rotation_deg=rot,
                tag=tag,
                final_loss=fin,
                loss_reduction_pct=red,
                coverage=float(cov),
                uniformity_median=float(t["median_peak_to_bg"]),
                uniformity_min=float(t["min_peak_to_bg"]),
                dominant_d_A=dom_d,
                dominant_d_std_A=dom_d_std,
                low_freq_frac=float(lf),
                edge_score=float(edge),
            )
        )

    csv_path = diag_root / "geometry_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(
            [
                "tag",
                "convention",
                "rotation_deg",
                "final_loss",
                "loss_reduction_pct",
                "coverage_tiles_frac",
                "uniformity_median_peak_bg",
                "uniformity_min_peak_bg",
                "dominant_d_A",
                "dominant_d_std_A",
                "low_freq_power_frac",
                "edge_artifact_score",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.tag,
                    r.convention,
                    f"{r.rotation_deg:.3g}",
                    f"{r.final_loss:.10g}",
                    f"{r.loss_reduction_pct:.6g}",
                    f"{r.coverage:.6g}",
                    f"{r.uniformity_median:.6g}",
                    f"{r.uniformity_min:.6g}",
                    f"{r.dominant_d_A:.6g}" if r.dominant_d_A is not None else "",
                    f"{r.dominant_d_std_A:.6g}" if r.dominant_d_std_A is not None else "",
                    f"{r.low_freq_frac:.6g}",
                    f"{r.edge_score:.6g}",
                ]
            )

    conventions = [
        "A_identity",
        "B_transpose",
        "C_flip_x",
        "D_flip_y",
        "E_transpose_flip_x",
        "F_transpose_flip_y",
        "G_flip_xy",
        "H_transpose_flip_xy",
    ]
    rotations = [73.0, 17.0, -73.0, -17.0]
    idx = {(c, r): (conventions.index(c), rotations.index(r)) for c in conventions for r in rotations}

    def map_metric(getter):
        out = {}
        for rr in rows:
            if (rr.convention, rr.rotation_deg) in idx:
                out[idx[(rr.convention, rr.rotation_deg)]] = float(getter(rr))
        return out

    heatmap(
        map_metric(lambda r: r.coverage),
        diag_root / "geometry_matrix_coverage_heatmap.png",
        "tile FFT periodicity coverage (4x4 tiles)",
        "magma",
    )
    heatmap(
        map_metric(lambda r: r.uniformity_min),
        diag_root / "geometry_matrix_uniformity_heatmap.png",
        "minimum tile FFT peak/background (4x4 tiles)",
        "magma",
    )

    montage_panels(panels, diag_root / "geometry_matrix_montage.png")

    ranked = sorted(
        rows,
        key=lambda r: (
            -r.coverage,
            -r.uniformity_min,
            -(0.0 if r.dominant_d_std_A is None else -r.dominant_d_std_A),
            -r.uniformity_median,
            r.low_freq_frac,
            r.edge_score,
            r.final_loss,
        ),
    )
    best = ranked[0] if ranked else None
    (diag_root / "geometry_matrix_summary.json").write_text(
        json.dumps(
            {
                "best": best.__dict__ if best else None,
                "top8": [r.__dict__ for r in ranked[:8]],
                "n_done": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

