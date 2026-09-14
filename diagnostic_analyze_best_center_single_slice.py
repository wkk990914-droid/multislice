from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diagnostic_analyze_single_slice import (
    fft_power,
    plot_uniformity,
    robust_clim,
    save_loss,
    save_single,
    spatial_uniformity,
)


def main() -> int:
    base_run = Path("runs/pyrex_atomic_phase_test")
    run_dir = base_run / "diagnostics" / "best_center_single_slice_run"
    out_dir = base_run / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = np.load(run_dir / "stages" / "single_slice" / "obj_cropped.npy")
    loss = np.load(run_dir / "stages" / "single_slice" / "loss_history.npy").astype(np.float64)
    img = np.asarray(obj[0], dtype=np.float64)

    vmin, vmax = robust_clim(img, 0.5, 99.5)
    save_single(
        img,
        out_dir / "best_center_single_slice_object.png",
        title="best-center single-slice object (potential/phase, 50 iters)",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        dpi=320,
    )

    p = fft_power(img)
    save_single(
        np.log1p(p),
        out_dir / "best_center_single_slice_fft.png",
        title="best-center single-slice FFT log(1+|F|^2)",
        cmap="inferno",
        dpi=320,
    )
    save_loss(loss, out_dir / "best_center_single_slice_loss.png")

    metrics, summary = spatial_uniformity(img, tiles=4)
    plot_uniformity(metrics, out_dir / "best_center_fft_uniformity.png", "best-center spatial FFT uniformity (4x4)")
    (out_dir / "best_center_single_slice_uniformity.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

