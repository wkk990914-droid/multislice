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
    tile_metrics,
)


def parse_theta(tag: str) -> float:
    m = re.match(r"theta([pm])(\d+\.\d+)", tag)
    if not m:
        raise ValueError(f"bad tag: {tag}")
    sign = 1.0 if m.group(1) == "p" else -1.0
    return sign * float(m.group(2))


def main() -> int:
    base_run = Path("runs/pyrex_atomic_phase_test")
    out_dir = base_run / "diagnostics"
    rot_dir = out_dir / "rotation_subrefine"

    records = []
    for d in sorted(rot_dir.glob("theta*")):
        metrics_path = d / "stages" / "single_slice" / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        loss = np.load(d / "stages" / "single_slice" / "loss_history.npy").astype(np.float64)
        obj = np.load(d / "stages" / "single_slice" / "obj_cropped.npy")
        img = np.asarray(obj[0], dtype=np.float64)

        init = float(m["initial_loss"])
        fin = float(m["final_loss"])
        red = 100.0 * (init - fin) / init if init != 0 else None
        theta = parse_theta(d.name)

        power = fft_power(img)
        f = fft_peak_metrics(power)
        t = tile_metrics(img, tiles=4)
        cov = lattice_coverage(t["peak_to_bg"], threshold=10.0)
        lf = low_freq_fraction(power, frac=0.12)
        edge = edge_artifact_score(img)

        records.append(
            {
                "theta_deg": theta,
                "iters": int(loss.size),
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

    csv_path = out_dir / "rotation_subrefine.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(
            [
                "theta_deg",
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
                    f"{r['theta_deg']:.2f}",
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

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thetas = np.array([r["theta_deg"] for r in ranked], dtype=np.float64)
    uni = np.array([r["uniformity_median"] for r in ranked], dtype=np.float64)
    cov = np.array([r["lattice_coverage"] for r in ranked], dtype=np.float64)
    loss = np.array([r["final_loss"] for r in ranked], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(8.4, 4.0), dpi=250)
    ax1.plot(thetas, uni, marker="o", label="uniformity median")
    ax1.plot(thetas, cov, marker="o", label="coverage")
    ax1.set_xlabel("theta (deg)")
    ax1.set_ylabel("metric (arb.)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(thetas, loss, marker="x", color="black", alpha=0.55, label="final loss")
    ax2.set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(out_dir / "rotation_subrefine.png", bbox_inches="tight")
    plt.close(fig)

    (out_dir / "rotation_subrefine_best.json").write_text(json.dumps({"best": best, "all": records}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

