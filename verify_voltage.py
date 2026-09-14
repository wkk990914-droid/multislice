"""Verify the accelerating voltage from the experimental diffraction patterns.

The Bragg disks sit at scattering angle 2*theta_B = lambda / d.  Measuring their
radius in mrad (we know dp_resolution = 1.2155 mrad/px from the file) and taking
the ratio of two reflections fixes the lattice *without* knowing the voltage;
the absolute radius then gives lambda, hence the voltage.
"""
import logging

import h5py
import numpy as np

logging.basicConfig(level=logging.WARNING)

PATH = "PTYREX_8_FOV5.0nm_def0nm_alpha25.0mrad_dx0.020nm_R256x256_Q256x256_CL0.00082m_ro73deg_trueCL87.h5"
DPR = 1.2154998186711752  # mrad per detector pixel

# Relativistic electron wavelength (Angstrom) as a function of energy in volts.
def wavelength_A(eV: float) -> float:
    m0c2 = 510998.95  # eV
    lam_nm = 1.226434 / np.sqrt(eV * (1 + eV / (2 * m0c2))) / 1e3 * 1e3
    # Standard: lambda[nm] = 1.226434 / sqrt(E) * 1/sqrt(1+E/1022000) with E in V,
    # but the relativistic correction uses 2*m0c2 in the denominator.
    lam_A = 12.26434 / np.sqrt(eV) / np.sqrt(1 + eV / (2 * m0c2)) / 10.0
    return lam_A


print("reference wavelengths (Angstrom):")
for E in (80e3, 100e3, 200e3, 300e3):
    print(f"  {E/1e3:5.0f} kV  lambda = {wavelength_A(E):.5f}")

with h5py.File(PATH, "r") as f:
    frames = f["data/frames"]

    # Average a block of neighbouring scan positions: the lattice is the same,
    # so Bragg disks add coherently while noise averages down.
    dp = frames[110:150, 110:150].mean(axis=(0, 1)).astype(np.float64)

print(f"\nmean DP {dp.shape}, max {dp.max():.4g}")

n = dp.shape[0]
cy = cx = n / 2 - 0.5
ys, xs = np.mgrid[0:n, 0:n]
rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

# Direct-beam disc radius from the file: BF_radius_pixels = 20.57 px (25 mrad).
core = rr < 22.0
outside = ~core

# Find local maxima outside the direct disc.
from scipy.ndimage import maximum_filter, label, center_of_mass

img = dp.copy()
img[core] = 0.0
mx = maximum_filter(img, size=5)
peaks = (img == mx) & (img > 0.15 * img.max())
py, px = np.nonzero(peaks)
rad = np.sqrt((py - cy) ** 2 + (px - cx) ** 2)
ang = np.degrees(np.arctan2(py - cy, px - cx)) % 360
strength = img[py, px]

order = np.argsort(-strength)
print(f"\n{len(py)} candidate Bragg disks outside the direct disc. Top 24:")
print(f"  {'r[px]':>7} {'2theta[mrad]':>12} {'angle[deg]':>10} {'strength':>9}")
for i in order[:24]:
    print(f"  {rad[i]:7.2f} {rad[i]*DPR:12.2f} {ang[i]:10.1f} {strength[i]:9.3g}")

# Cluster by radius -- hexagonal symmetry gives 6 or 12 equivalent disks.
sel = order[:40]
radii = np.sort(rad[sel])
print("\nradius clusters (px):")
clusters = []
for r in radii:
    if clusters and abs(clusters[-1][-1] - r) < 1.5:
        clusters[-1].append(r)
    else:
        clusters.append([r])
for c in clusters:
    rm = float(np.mean(c))
    print(f"  r = {rm:6.2f} px  -> 2theta = {rm*DPR:7.3f} mrad   (n={len(c)})")

# --- invert for the lattice / voltage ------------------------------------- #
# For a hexagonal lattice the first two shells are at d = a and a/sqrt(3),
# so 2theta ratios must be sqrt(3) : 1.
print("\n=== hexagonal shell test (d=a and d=a/sqrt(3) -> ratio sqrt3=1.732) ===")
shell_r = [float(np.mean(c)) * DPR for c in clusters if len(c) >= 3]
for i in range(len(shell_r)):
    for j in range(i + 1, len(shell_r)):
        ratio = shell_r[j] / shell_r[i]
        if abs(ratio - np.sqrt(3)) < 0.08:
            print(
                f"  pair {shell_r[i]:.2f} / {shell_r[j]:.2f} mrad ratio={ratio:.3f} "
                f"-> consistent with a hexagonal lattice"
            )

print("\n=== voltage implied by each cluster, assuming WSe2 a = 3.28 A ===")
for c in clusters:
    rm = float(np.mean(c)) * DPR
    for label_d, d in (("a=3.28 (10-10)", 3.28), ("a/sqrt3=1.89 (11-20)", 1.894),
                       ("c=6.50 (0002)", 6.50)):
        lam = rm / 1e3 * d  # lambda[A] = 2theta[rad] * d[A]
        if 0.01 < lam < 0.08:
            # invert wavelength -> energy
            lo, hi = 20e3, 500e3
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if wavelength_A(mid) > lam:
                    lo = mid
                else:
                    hi = mid
            E = 0.5 * (lo + hi)
            print(f"  2theta={rm:7.3f} mrad  {label_d:20s} lambda={lam:.5f} A -> {E/1e3:6.1f} kV")
