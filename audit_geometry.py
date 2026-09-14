"""Measure the actual probe extent and redo the lattice FFT correctly.

Two things the earlier analysis got wrong:
  * the object pixel size is the *post-preprocess* sampling (0.1789 A), not the
    raw scan step (0.1953 A);
  * the DC mask radius was large enough to delete the real lattice peaks.
"""
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

from ptycho.analysis import decompose_object, to_numpy  # noqa: E402
from ptycho.config import PipelineConfig  # noqa: E402
from ptycho.worker import load_checkpoint, prepare_dataset  # noqa: E402

OUT = Path("runs/pyrex_atomic_phase_test")
cfg = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")
_d, pdset = prepare_dataset(cfg)
pty = load_checkpoint(OUT, "dip_free_extra", dset=pdset, device="cpu")

samp = float(np.asarray(pty.sampling).ravel()[0])
lam = 12.26434 / np.sqrt(cfg.experiment.probe_energy) / np.sqrt(
    1 + cfg.experiment.probe_energy / (2 * 510998.95)
) / 10.0
scan_n = cfg.data.scan_crop[1] - cfg.data.scan_crop[0]

print("=" * 70)
print("1. PROBE EXTENT  (decides ptychographic overlap)")
print("=" * 70)
pr = to_numpy(pty.probe)                      # (num_probes, y, x)
inten = np.abs(pr) ** 2
inten = inten / inten.max()
n = inten.shape[-1]
cy = cx = n / 2 - 0.5
ys, xs = np.mgrid[0:n, 0:n]
rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

# Radius enclosing 95% of the probe current.
prof = []
for rad in np.arange(1, n / 2):
    m = rr <= rad
    prof.append((rad, inten[:, m].sum()))
prof = np.array(prof)
tot = prof[:, 1].max()
r95 = prof[prof[:, 1] >= 0.95 * tot][0, 0]
r80 = prof[prof[:, 1] >= 0.80 * tot][0, 0]
print(f"probe array             {n} x {n} px  ({n*samp:.1f} A field)")
print(f"95% current radius      {r95:.0f} px = {r95*samp:.2f} A  -> diameter {2*r95*samp:.2f} A")
print(f"80% current radius      {r80:.0f} px = {r80*samp:.2f} A")
print(f"scan region             {scan_n} px = {scan_n*samp:.2f} A")
print(f"object grid (full)      {pty.obj.shape[-1]} px = {pty.obj.shape[-1]*samp:.2f} A")
print(f"probe diam / scan FOV   {2*r95*samp/(scan_n*samp):.2f}")
free = scan_n * samp - 2 * r95 * samp
print(f"free probe travel       {free:.2f} A  ({free/(2*r95*samp):.2f} x probe diameter)")
print("NOTE: overlap >~50% is required for a well-posed ptychography problem.")

print()
print("=" * 70)
print("2. LATTICE FFT OF THE PROJECTED PHASE (corrected sampling + mask)")
print("=" * 70)
obj = np.load(f"{OUT}/stages/dip_free_extra/obj_cropped.npy")
phase, _, kind = decompose_object(obj)
proj = phase.sum(axis=0)
ny, nx = proj.shape
print(f"projected phase grid    {ny} x {nx} px = {ny*samp:.2f} x {nx*samp:.2f} A")
print(f"pixel size used         {samp:.4f} A  (was wrongly {cfg.experiment.scan_step_size:.4f} A)")

img = proj - proj.mean()
win = np.outer(np.hanning(ny), np.hanning(nx))
F = np.fft.fftshift(np.fft.fft2(img * win))
P = np.abs(F) ** 2
cy2, cx2 = ny // 2, nx // 2
yy, xx = np.mgrid[0:ny, 0:nx]
r = np.sqrt((yy - cy2) ** 2 + (xx - cx2) ** 2)
dq = 1.0 / (ny * samp)

A = 3.28
shells = {"10-10 (d=3.28)": A, "11-20 (d=1.89)": A / np.sqrt(3),
          "20-20 (d=1.64)": A / 2, "21-10 (d=1.40)": A / np.sqrt(7) * 1.0}
print(f"\nfrequency scale dq      {dq:.5f} A^-1 / px")
for name, d in shells.items():
    rp = 1.0 / d / dq
    print(f"  {name:16s} expected at r = {rp:5.2f} px")

# Background annulus well outside the lattice region.
ann = (r > 12) & (r < 40)
bg = np.median(P[ann])
print(f"\nbackground (r 12-40 px) {bg:.4g}")

print("\npeak power inside each expected shell (+-1.5 px), relative to background:")
for name, d in shells.items():
    rp = 1.0 / d / dq
    band = np.abs(r - rp) < 1.5
    if band.sum() == 0:
        continue
    mx = P[band].max()
    print(f"  {name:16s} max/bg = {mx/bg:8.1f}   mean/bg = {P[band].mean()/bg:6.2f}")

# Are there 6-fold symmetric peaks at the 3.28 A spacing?
rp = 1.0 / A / dq
band = np.abs(r - rp) < 1.5
vals = P.copy()
vals[~band] = 0
from scipy.ndimage import maximum_filter

mx = maximum_filter(vals, size=5)
pk = (vals == mx) & band & (vals > 5 * bg)
py, px = np.nonzero(pk)
if len(py):
    ang = np.degrees(np.arctan2(py - cy2, px - cx2)) % 360
    o = np.argsort(-P[py, px])
    print(f"\n{len(py)} peaks near the (10-10) ring, strongest:")
    for i in o[:8]:
        print(f"   angle {ang[i]:6.1f} deg  r={r[py[i],px[i]]:5.2f} px  power={P[py[i],px[i]]/bg:7.1f}x bg")
    print("  a hexagonal lattice should show 6 peaks ~60 deg apart")
else:
    print("\nNO peaks found near the (10-10) ring -> lattice periodicity absent")

print()
print("=" * 70)
print("3. PHASE MAGNITUDE vs PHYSICS")
print("=" * 70)
print(f"per-slice phase     max {phase.max():.4f} rad   std {phase.std():.4f}")
print(f"projected phase     max {proj.max():.4f} rad   std {proj.std():.4f}")
print(f"16 slices x {phase.max():.4f} = {16*phase.max():.3f} rad")
print("\nExpected WSe2 [0001] column phase at configured beam energy, 1 A slice:")
# sigma_E = pi/(lambda*E) in rad/V/A ; V0 ~ 7-9 V.A for WSe2 columns
m0c2 = 510998.95
E0 = cfg.experiment.probe_energy
sigma = np.pi * (E0 + m0c2) / (lam * E0 * (E0 + 2 * m0c2) / m0c2) * 1e-10  # rad/(V.A)
sigma = np.pi / (lam * 1e-10) * (1 / (E0 * (1 + E0 / (2 * m0c2)) / m0c2 * 1e6)) * 1e-6
V0 = 7.5  # V.A per Angstrom column, order of magnitude for WSe2
print(f"  sigma_E ~ {sigma:.4g} rad/(V.A) -> phase/slice ~ {sigma*V0:.3f} rad")
print(f"  observed/slice max = {phase.max():.4f} rad")
print(f"  ratio observed/expected = {phase.max()/(sigma*V0):.2f}")
