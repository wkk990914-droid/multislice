from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if np.isnan(xf):
        return "n/a"
    return f"{xf:.6g}"


def main() -> int:
    base_run = Path("runs/pyrex_atomic_phase_test")
    diag = base_run / "diagnostics"

    dp = _read_json(diag / "dp_center_diagnostics.json")
    coarse = _read_json(diag / "center_sweep_summary.json")
    best_center = _read_json(diag / "BEST_DP_CENTER.json")
    best_rot = _read_json(diag / "BEST_ROTATION.json")
    best_center_uni = _read_json(diag / "best_center_single_slice_uniformity.json")

    center_geom = dp["center_geom"]
    center_com = dp["center_com"]
    center_fit = dp["center_fit"]
    best_dp = best_center["BEST_DP_CENTER_Q256"]
    best_dp_off = best_center["BEST_DP_CENTER_offset_from_geom"]
    best_dp_crop = best_center["BEST_DP_CROP"]

    theta_best = best_rot["BEST_ROTATION"]["theta_deg"] if best_rot.get("BEST_ROTATION") else None

    lines: list[str] = []
    lines.append("# GEOMETRY_REPAIR_REPORT")
    lines.append("")
    lines.append("This report covers controlled detector-center + rotation diagnostics using **SINGLE-SLICE** only.")
    lines.append("It does **not** justify multislice restart yet unless the single-slice gate passes.")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    for p in (
        diag / "dp_center_diagnostics.png",
        diag / "center_sweep_heatmap_loss.png",
        diag / "center_sweep_heatmap_uniformity.png",
        diag / "center_sweep_heatmap_coverage.png",
        diag / "center_sweep_montage.png",
        diag / "fine_center_heatmap_loss.png",
        diag / "fine_center_heatmap_uniformity.png",
        diag / "fine_center_heatmap_coverage.png",
        diag / "fine_center_montage.png",
        diag / "best_center_single_slice_object.png",
        diag / "best_center_single_slice_fft.png",
        diag / "best_center_single_slice_loss.png",
        diag / "best_center_fft_uniformity.png",
        diag / "fine_rotation_comparison.png",
        diag / "rotation_subrefine.png",
    ):
        if p.exists():
            lines.append(f"- {p.resolve()}")
    lines.append("")

    lines.append("## 1) Original DP center (geometric)")
    lines.append("")
    lines.append(f"- center_geom = (cx={_fmt(center_geom['cx'])}, cy={_fmt(center_geom['cy'])}) in Q256 coords")
    lines.append("")

    lines.append("## 2) BF CoM center (from mean DP)")
    lines.append("")
    lines.append(f"- center_com  = (cx={_fmt(center_com['cx'])}, cy={_fmt(center_com['cy'])})")
    lines.append(
        f"- offset_com_from_geom = (dx={_fmt(dp['offset_com_from_geom']['dx'])}, dy={_fmt(dp['offset_com_from_geom']['dy'])}) px"
    )
    lines.append("")

    lines.append("## 3) Disc-fit / radial-symmetry center")
    lines.append("")
    lines.append(f"- center_fit  = (cx={_fmt(center_fit['cx'])}, cy={_fmt(center_fit['cy'])})")
    lines.append(
        f"- offset_fit_from_geom = (dx={_fmt(dp['offset_fit_from_geom']['dx'])}, dy={_fmt(dp['offset_fit_from_geom']['dy'])}) px"
    )
    lines.append(f"- BF radius (expected) = {_fmt(dp['bf_radius_px_expected'])} px; (fit) = {_fmt(dp['bf_radius_px_fit'])} px")
    lines.append("")

    lines.append("## 4) Coarse center sweep result (±2 px around BF CoM)")
    lines.append("")
    lines.append("- Sweep grid: dx,dy ∈ {-2,-1,0,+1,+2} relative to center_com; 18 iters each; fixed rotation θ=17°; fixed seed.")
    lines.append(f"- Best coarse candidate (by coverage→uniformity ranking): dx={coarse['best']['dx']}, dy={coarse['best']['dy']}")
    lines.append(
        f"  - lattice coverage (tiles) = {_fmt(coarse['best']['lattice_coverage'])}, uniformity median/min = "
        f"{_fmt(coarse['best']['uniformity_median'])} / {_fmt(coarse['best']['uniformity_min'])}"
    )
    lines.append("")

    lines.append("## 5) Best center (after fine integer refinement)")
    lines.append("")
    lines.append(
        f"- Best center in Q256 coords: BEST_DP_CENTER = (cx={_fmt(best_dp['cx'])}, cy={_fmt(best_dp['cy'])})"
    )
    lines.append(f"- Required center shift from geometric: (dx={_fmt(best_dp_off['dx'])}, dy={_fmt(best_dp_off['dy'])}) px")
    lines.append(f"- Implemented as symmetric 192×192 crop: dp_crop = {best_dp_crop} (qy0,qy1,qx0,qx1)")
    lines.append("")

    lines.append("## 6) Best rotation (with BEST_DP_CENTER fixed)")
    lines.append("")
    lines.append(f"- BEST_ROTATION (coarse) = {_fmt(theta_best)} degrees")
    lines.append("- See: fine_rotation.csv and fine_rotation_comparison.png")
    lines.append("")

    lines.append("## 7) Beam energy")
    lines.append("")
    lines.append("- SOURCE_CONFIRMED_ENERGY = UNKNOWN (H5 has no energy/voltage attrs; no separate simulation config found).")
    lines.append("- SAMPLING_INFERRED_ENERGY = UNKNOWN/UNRELIABLE (no clear Bragg peaks outside BF edge; inference collapses to BF-edge artifacts).")
    lines.append("- ENERGY_USED_FOR_DIAGNOSTIC = 80 keV (from configs/pyrex_atomic_phase.yaml).")
    lines.append("")

    lines.append("## 8) Single-slice gate: FAIL → PASS ?")
    lines.append("")
    lines.append("- Best-center single-slice (50 iters, fresh init) tile FFT-uniformity summary:")
    lines.append(
        f"  - median peak/bg = {_fmt(best_center_uni['median_peak_to_background'])}, "
        f"min = {_fmt(best_center_uni['min_peak_to_background'])}, max = {_fmt(best_center_uni['max_peak_to_background'])}"
    )
    lines.append("- Gate decision: **STILL FAIL** (at least one tile shows no detectable periodic FFT peaks).")
    lines.append("")

    lines.append("## 9) Final decision")
    lines.append("")
    lines.append("1. Original DP center = center_geom")
    lines.append("2. BF CoM center = center_com")
    lines.append("3. Best center = BEST_DP_CENTER")
    lines.append("4. Required center shift = BEST_DP_CENTER - center_geom")
    lines.append("5. Best rotation = BEST_ROTATION")
    lines.append("6. Beam energy = UNKNOWN (80 keV assumed for diagnostics only)")
    lines.append("7. Did single-slice change from FAIL to PASS? **NO**")
    lines.append("8. Fraction of FOV with consistent lattice contrast: insufficient (tile metric min hits 0)")
    lines.append("9. Tile-wise FFT confirms global periodicity? **NO** (non-uniform)")
    lines.append("10. Is it justified to run full multislice? **NO**")
    lines.append("")

    (base_run / "GEOMETRY_REPAIR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

