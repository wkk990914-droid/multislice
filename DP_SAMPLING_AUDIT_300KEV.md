# 300 keV DP Sampling Audit + Single-Slice Screening

This workflow generates a diffraction-plane calibration audit from the raw PYREX H5 and then runs **single-slice** ptychography screening jobs on **GPU 5/6/7 only**.

## Inputs

- Raw H5:
  - `/public/home/wangkai/shiyunZhang/pyrex/PTYREX_8_FOV5.0nm_def0nm_alpha25.0mrad_dx0.020nm_R256x256_Q256x256_CL0.00082m_ro73deg_trueCL87.h5`
- Physics:
  - Beam energy: 300 keV
  - Probe semi-angle: 25 mrad

## Outputs

All outputs go to:

- `runs/dp_sampling_audit_300kev/`

## Step 1 — Build the mean DP + BF radius audit

```bash
python dp_sampling_audit_300kev.py
```

Key outputs:

- `runs/dp_sampling_audit_300kev/00_mean_dp.npy`
- `runs/dp_sampling_audit_300kev/00_mean_dp_raw.png`
- `runs/dp_sampling_audit_300kev/00_mean_dp_log.png`
- `runs/dp_sampling_audit_300kev/01_bf_center_detection.png`
- `runs/dp_sampling_audit_300kev/02_radial_profile.png`
- `runs/dp_sampling_audit_300kev/03_bf_edge_fit.png`
- `runs/dp_sampling_audit_300kev/04_dp_crop_192_overlay.png`
- `runs/dp_sampling_audit_300kev/dp_sampling_audit.json`

The JSON includes:

- BF center (geom/CoM/fit) in pixels
- BF radius from multiple estimators + uncertainty
- Calibrated `dp_sampling = alpha / R_BF_px`
- Optional metadata dp-sampling (if present in H5 attrs)
- Crop theta_max checks and a recommended dp-crop size
- PASS/WARN/FAIL calibration status

## Step 2 — Single-slice screening (300 keV)

```bash
python single_slice_screening_300kev.py
```

This generates a small parameter matrix around:

- dp_sampling: metadata vs calibrated
- dp_crop: 192 vs recommended
- rotation: 73° plus ±1°/±2°

and runs independent single-slice jobs with fixed seed and fixed hyperparameters, scheduled across GPU 5/6/7.

Key outputs:

- `runs/dp_sampling_audit_300kev/screening_summary.csv`
- `runs/dp_sampling_audit_300kev/screening_outputs/*__phase.png`
- `runs/dp_sampling_audit_300kev/screening_outputs/*__phase_fft.png`
- `runs/dp_sampling_audit_300kev/screening_outputs/*__loss_curve.png`

Notes:

- With `obj_type=potential`, the reconstructed object is real (phase/potential). The saved `amplitude` panel is `|exp(i·phase)|` and will be ~1 everywhere; it is included only for completeness.

## Step 3 — Automated ranking + montages

```bash
python analyze_single_slice_screening_300kev.py
```

Outputs:

- `runs/dp_sampling_audit_300kev/05_screening_ranking.png`
- `runs/dp_sampling_audit_300kev/FINAL_single_slice_screening_montage.png`
- `runs/dp_sampling_audit_300kev/FINAL_fft_screening_montage.png`
- `runs/dp_sampling_audit_300kev/best_run.json`

