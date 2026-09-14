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


def montage(paths: list[Path], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs = [plt.imread(p) for p in paths]
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), dpi=300)
    axes = np.asarray(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(imgs[i])
            ax.set_title(paths[i].name.replace("__phase.png", ""), fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def ranking_plot(summary_csv: Path, out_path: Path) -> None:
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_csv)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    df["rank_score"] = (
        10.0 * df["atomic_periodicity_score"].astype(float)
        + 0.2 * df["tile_min_peak_bg"].astype(float)
        + 0.05 * df["fft_peak_prominence"].astype(float)
        - 0.0001 * df["final_loss"].astype(float)
    )
    df = df.sort_values("rank_score", ascending=False)
    fig, ax = plt.subplots(figsize=(10.2, 4.4), dpi=300)
    ax.bar(np.arange(len(df)), df["rank_score"].values)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["run_id"].values, rotation=90, fontsize=6)
    ax.set_ylabel("composite score")
    ax.set_title("single-slice screening ranking (300 keV)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    best = df.iloc[0].to_dict()
    (out_path.parent / "best_run.json").write_text(json.dumps(best, indent=2), encoding="utf-8")


def main() -> int:
    root = Path("runs/dp_sampling_audit_300kev")
    out_img = root / "screening_outputs"
    csv_path = root / "screening_summary.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}")

    phase_paths = sorted(out_img.glob("*__phase.png"))
    fft_paths = sorted(out_img.glob("*__phase_fft.png"))
    if phase_paths:
        montage(phase_paths, root / "FINAL_single_slice_screening_montage.png", "phase/potential panels")
    if fft_paths:
        montage(fft_paths, root / "FINAL_fft_screening_montage.png", "FFT panels")

    ranking_plot(csv_path, root / "05_screening_ranking.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

