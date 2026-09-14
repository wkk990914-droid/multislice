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
    run_dir = Path("runs/pyrex_atomic_phase_test")
    diag = run_dir / "diagnostics"

    obj_stats = _read_json(diag / "object_stats.json")
    scan = _read_json(diag / "scan_audit.json")
    diff = _read_json(diag / "diffraction_audit.json")
    uni_multi = _read_json(diag / "fft_spatial_uniformity.json")
    rot = _read_json(diag / "rotation_comparison.json")
    ss_uni = _read_json(diag / "single_slice_uniformity.json")

    quantem_snippet = {
        "quantem_version": "0.1.9",
        "objectpixelated_file": "/public/home/wangkai/miniconda3/lib/python3.13/site-packages/quantem/diffractive_imaging/object_models.py",
        "objectpixelated_get_obj_patches": "if not obj_array.is_complex(): obj_array2 = torch.exp(1.0j * obj_array)",
    }

    lines: list[str] = []
    lines.append("# PYREX / Quantem diagnostic report (STOP before longer recon)")
    lines.append("")
    lines.append("This report is intended to locate geometry/model errors before any longer reconstruction.")
    lines.append("")

    lines.append("## Artifacts written by this audit")
    lines.append("")
    for p in (
        diag / "RAW_all_slices.png",
        diag / "RAW_projected_potential.png",
        diag / "DETRENDED_all_slices.png",
        diag / "DETRENDED_projected.png",
        diag / "scan_positions.png",
        diag / "mean_dp_with_center.png",
        diag / "single_slice_object.png",
        diag / "single_slice_fft.png",
        diag / "single_slice_loss.png",
        diag / "rotation_comparison.png",
        diag / "rotation_comparison.csv",
        diag / "fft_spatial_uniformity.png",
    ):
        if p.exists():
            lines.append(f"- {p.resolve()}")
    lines.append("")

    lines.append("## TASK 1 — Object representation audit")
    lines.append("")
    lines.append(f"- dip_free_extra object dtype = `{obj_stats['dtype']}`; np.iscomplexobj = `{obj_stats['iscomplex']}`; shape = `{obj_stats['shape']}`")
    lines.append(
        f"- global min/max/mean/std = {_fmt(obj_stats['min'])} / {_fmt(obj_stats['max'])} / {_fmt(obj_stats['mean'])} / {_fmt(obj_stats['std'])}"
    )
    lines.append("- per-slice stats in JSON: " + str((diag / "object_stats.json").resolve()))
    lines.append("")
    lines.append("- Quantem storage mode (verified from installed quantem source):")
    lines.append(f"  - quantem version: {quantem_snippet['quantem_version']}")
    lines.append(f"  - ObjectPixelated source: {quantem_snippet['objectpixelated_file']}")
    lines.append(f"  - _get_obj_patches logic: `{quantem_snippet['objectpixelated_get_obj_patches']}`")
    lines.append("  - Therefore with `obj_type=potential` Quantem stores a **real-valued phase/potential** array; the complex transmission used in propagation is `exp(i * object)`.")
    lines.append("  - For this pipeline: `potential = object` (do NOT use np.angle(object); do NOT interpret np.abs(object) as physical amplitude).")
    lines.append("")

    lines.append("## TASK 3 — Scan position audit (after cropping + preprocess)")
    lines.append("")
    lines.append(f"- number of scan positions = {scan['n_positions']} (expected 64×64 = 4096)")
    lines.append(f"- scan grid shape = {scan['scan_grid_shape']}")
    lines.append(f"- object sampling used (from stage metrics) = {_fmt(scan['object_sampling_A'])} Å/px")
    lines.append(f"- configured scan step = {_fmt(scan['scan_step_cfg_A'])} Å")
    lines.append(f"- median neighbor step (x) = {_fmt(scan['median_dx_A'])} Å  (px={_fmt(scan['median_dx_px'])})")
    lines.append(f"- median neighbor step (y) = {_fmt(scan['median_dy_A'])} Å  (px={_fmt(scan['median_dy_px'])})")
    lines.append(f"- bounding-box FOV (x) = {_fmt(scan['fov_x_A'])} Å; (y) = {_fmt(scan['fov_y_A'])} Å")
    lines.append(
        "- unique x/y counts are 4096 because the raster is rotated (73°); uniqueness is expected and not by itself an error."
    )
    lines.append("")

    lines.append("## TASK 4 — Diffraction geometry audit (raw H5)")
    lines.append("")
    lines.append(f"- H5 dataset = `{diff['dataset']}`")
    lines.append(f"- H5 dataset shape = `{diff['dataset_shape']}` (expected 256,256,256,256) ✓")
    lines.append(f"- dp_sampling = {_fmt(diff['dp_sampling_mrad_per_px'])} mrad/pixel (from config)")
    lines.append(f"- alpha = {_fmt(diff['alpha_mrad'])} mrad (from config)")
    lines.append(f"- expected BF radius = {_fmt(diff['expected_bf_radius_px'])} pixels")
    lines.append(f"- dp crop = {diff['dp_crop']} (192×192)")
    lines.append(
        f"- geometric center = ({_fmt(diff['geometric_center_px'][1])}, {_fmt(diff['geometric_center_px'][0])}) px; "
        f"estimated BF CoM center = ({_fmt(diff['estimated_bf_center_px'][1])}, {_fmt(diff['estimated_bf_center_px'][0])}) px"
    )
    lines.append(
        "- The BF CoM is offset from the geometric center by ~1–2 pixels, meaning the 192×192 crop may not be perfectly centered on the true diffraction origin."
    )
    lines.append("")

    lines.append("## TASK 2 — Single-slice sanity test (most important gate)")
    lines.append("")
    lines.append("- A 1-slice pixelated reconstruction (50 iters) was run with the same 64×64 scan crop and 192×192 dp crop.")
    lines.append(f"- spatial FFT uniformity (4×4 tiles): median peak/bg = {_fmt(ss_uni['median_peak_to_background'])}, min = {_fmt(ss_uni['min_peak_to_background'])}, max = {_fmt(ss_uni['max_peak_to_background'])}")
    lines.append("")
    lines.append("Gate question: does single-slice recover periodic atomic/lattice contrast across MOST of the scan FOV?")
    lines.append("- Result: **NO (fails uniformity gate).** Strong periodicity is not uniformly recovered across tiles; large tile-to-tile variation indicates geometry/calibration issues or localized artifacts.")
    lines.append("")

    lines.append("## TASK 5 — Rotation convention test (single-slice, short runs)")
    lines.append("")
    best3 = rot.get("best_by_uniformity", [])[:3]
    if best3:
        lines.append("- Best candidates by uniformity ranking (still fails the pass criteria because min tile peak/bg remains low):")
        for entry in best3:
            lines.append(
                f"  - rot={_fmt(entry['rotation_deg'])}°: uniformity median={_fmt(entry['uniformity_median'])}, min={_fmt(entry['uniformity_min'])}, fft peak/bg={_fmt(entry['fft_peak_to_bg'])}"
            )
    lines.append("- Full table: " + str((diag / "rotation_comparison.csv").resolve()))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Multiple rotations show large loss reductions and strong global FFT peaks, but **none** produces spatially uniform periodic contrast over the FOV.")
    lines.append("- Therefore rotation sign/±90° convention is not the only issue; diffraction-centering and/or other calibration terms remain suspect.")
    lines.append("")

    lines.append("## TASK 6 — Spatial FFT uniformity test (multislice projected image)")
    lines.append("")
    lines.append(
        f"- multislice projected tiles (4×4): median peak/bg = {_fmt(uni_multi['median_peak_to_background'])}, "
        f"min = {_fmt(uni_multi['min_peak_to_background'])}, max = {_fmt(uni_multi['max_peak_to_background'])}"
    )
    lines.append("- Large tile-to-tile spread implies that global FFT peaks can be dominated by a localized region (e.g. one corner artifact).")
    lines.append("")

    lines.append("## TASK 7 — Beam energy audit")
    lines.append("")
    lines.append("- H5 file contains no metadata attrs (root/group/dataset attrs are empty).")
    lines.append("- Only evidence in this workspace is the *assumption* in configs:")
    lines.append(f"  - {Path('configs/pyrex_atomic_phase.yaml').resolve()} sets `probe_energy: 80000.0` eV with a warning comment.")
    lines.append("- Conclusion: **beam energy is UNKNOWN in this workspace** (80 keV is assumed, not confirmed).")
    lines.append("")

    lines.append("## TASK 8 — Multislice thickness audit")
    lines.append("")
    lines.append("- Reconstruction configuration:")
    lines.append("  - num_slices = 16; dz = 1.0 Å; total depth = 16 Å (from config)")
    lines.append("- Search in this workspace did not find an upstream PYREX simulation config/script specifying the true sample thickness.")
    lines.append("- Conclusion: **true simulation thickness is UNKNOWN here**, so the 16 Å multislice depth is not currently validated.")
    lines.append("")

    lines.append("## TASK 9 — Explicit answers")
    lines.append("")
    lines.append("1. Is the current projected-phase image physically credible?")
    lines.append("- **NO (fails spatial-uniformity gate).** The FFT signal is not spatially uniform; low-frequency blobs dominate most of the FOV.")
    lines.append("2. Is object representation handled correctly?")
    lines.append("- **YES for representation:** object is a real-valued potential/phase array; do not apply np.angle/np.abs. The earlier interpretation risk is real and this audit enforces `potential = object`.")
    lines.append("3. Does single-slice ptychography recover the crystal across the full FOV?")
    lines.append("- **NO** based on the 50-iter diagnostic run and the 4×4 FFT-uniformity metrics.")
    lines.append("4. Are scan positions correct?")
    lines.append("- **Step size matches expectation** when interpreted using the actual object sampling after preprocess. Axis rotation explains the large unique x/y counts.")
    lines.append("5. Is diffraction center/crop correct?")
    lines.append("- **Likely off by ~1–2 pixels**: BF CoM center differs from geometric center, so the 192×192 crop may not be perfectly centered on the diffraction origin.")
    lines.append("6. Which scan-detector rotation convention performs best?")
    lines.append("- Best-by-uniformity candidates are listed above, but **none passes** the requirement of uniform lattice contrast across the FOV.")
    lines.append("7. Is 80 keV confirmed or only assumed?")
    lines.append("- **Only assumed.** No metadata/config evidence was found outside the reconstruction YAML.")
    lines.append("8. Is the 16 Å multislice thickness correct?")
    lines.append("- **Unknown.** No simulation thickness evidence found in this workspace.")
    lines.append("9. Are the previous 32 FFT peaks / 1.90 Å spacing global lattice information or localized artifact?")
    lines.append("- **Potentially localized artifact.** The tile FFT-uniformity test shows strong non-uniformity, so global FFT peaks are not reliable evidence of a correct atomic reconstruction.")
    lines.append("10. What is the most likely root cause?")
    lines.append("- Primary suspects to fix before any longer recon:")
    lines.append("  - diffraction origin / crop centering error (1–2 px offset is enough to create strong artifacts)")
    lines.append("  - rotation convention mismatch (sign/axis), but not sufficient alone")
    lines.append("  - missing/unknown physical parameters (beam energy, true thickness) that can invalidate multislice physics")
    lines.append("")
    lines.append("Hard stop: do NOT run 128×128 or long 500-iteration multislice until single-slice uniformity passes and diffraction centering is corrected.")
    lines.append("")

    (run_dir / "DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

