from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> int:
    out_dir = Path("runs/dp_sampling_audit_300kev")
    audit = json.loads((out_dir / "dp_sampling_audit.json").read_text(encoding="utf-8"))
    q = np.load(out_dir / "00_mean_dp.npy").astype(np.float64)

    center = audit["centers_px"]["fit"]
    cx, cy = float(center[0]), float(center[1])
    r = float(audit["bf_radius_fit_px"])

    dp_sampling = 1.2155
    alpha_check = r * dp_sampling
    rel_err = abs(alpha_check - 25.0) / 25.0 * 100.0
    status = "PASS" if rel_err < 2.0 else "WARN" if rel_err < 5.0 else "FAIL"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
    ax.imshow(np.log1p(q), cmap="inferno", origin="lower")
    ax.plot(cx, cy, marker="o", ms=6, mew=2, mfc="none", color="white", label="BF center (fit)")
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="white", lw=1.5, alpha=0.95))

    text = (
        f"Q256 BF geometry sanity check\\n"
        f"R_BF = {r:.4f} px\\n"
        f"dp_sampling = {dp_sampling:.4f} mrad/px\\n"
        f\"alpha_check = R*dp = {alpha_check:.3f} mrad\\n\"
        f\"rel_error = {rel_err:.3f}%  →  {status}\"
    )
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 6, "edgecolor": "none"},
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Mean DP (log1p) + fitted BF circle", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "Q256_BF_geometry_check.png", bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

