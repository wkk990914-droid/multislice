"""Diagnose why the loss plateaus: geometry, overlap, and lattice content."""
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)
from quantem.core import config as qconfig  # noqa: E402

from ptycho.analysis import (  # noqa: E402
    decompose_object,
    projected_phase_unwrapped,
    to_numpy,
)
from ptycho.config import PipelineConfig  # noqa: E402
from ptycho.worker import load_checkpoint, prepare_dataset  # noqa: E402

OUT = Path("runs/pyrex_atomic_phase_test")
cfg = PipelineConfig.from_yaml("configs/pyrex_atomic_phase.yaml")

_d, pdset = prepare_dataset(cfg)
pty = load_checkpoint(OUT, "dip_free_extra", dset=pdset, device="cpu")

print("=== geometry ===")
for attr in ("sampling", "dp_sampling", "num_probes", "probe_num_pixels_x",
             "probe_num_pixels_y", "num_dp_pixels_x", "num_dp_pixels_y"):
    try:
        print(f"  {attr:22s} {getattr(pty, attr)}")
    except Exception as e:
        print(f"  {attr:22s} <{type(e).__name__}>")

try:
    print("  shape (obj)          ", tuple(pty.obj.shape))
    print("  shape (probe)        ", tuple(pty.probe.shape))
except Exception as e:
    print("  shape unavailable:", e)

obj = to_numpy(pty.obj_cropped)
phase, amp, kind = decompose_object(obj)
proj = projected_phase_unwrapped(phase)

samp = float(np.asarray(pty.sampling).ravel()[0])
ny, nx = proj.shape
print(f"\n=== real-space grid ===")
print(f"  object pixel size    {samp:.4f} A")
print(f"  object extent        {ny*samp:.2f} x {nx*samp:.2f} A")
print(f"  WSe2 a=3.28 A ->     {ny*samp/3.28:.2f} x {nx*samp/3.28:.2f} unit cells")

print("\n=== phase statistics ===")
print(f"  per-slice  min {phase.min():.4f}  max {phase.max():.4f}  std {phase.std():.4f}")
print(f"  projected  min {proj.min():.4f}  max {proj.max():.4f}  std {proj.std():.4f}")
print(f"  peak/mean  {(proj.max()-proj.mean())/proj.std():.2f} sigma above mean")

# ---- power spectrum with explicit peak list ------------------------------- #
img = proj - proj.mean()
win = np.outer(np.hanning(ny), np.hanning(nx))
F = np.fft.fftshift(np.fft.fft2(img * win))
P = np.abs(F) ** 2
cy, cx = ny // 2, nx // 2
yy, xx = np.mgrid[0:ny, 0:nx]
r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
dq = 1.0 / (ny * samp)  # A^-1 per pixel

ring = (r > 2) & (r < 0.45 * min(ny, nx))
bg = np.median(P[ring])
print(f"\n=== power spectrum (projected phase) ===")
print(f"  freq scale           {dq*10:.4f} A^-1 / 10 px")
print(f"  max resolvable d     {1.0/(0.45*min(ny,nx)*dq):.2f} A")
print(f"  WSe2(100) d=3.28 ->  r={1/3.28/dq:.1f} px ; (110) d=1.89 -> r={1/1.894/dq:.1f} px")

# local maxima
from scipy.ndimage import maximum_filter
mx = maximum_filter(P, size=7)
peaks = (P == mx) & ring & (P > 8 * bg)
ys, xs = np.nonzero(peaks)
if len(ys):
    strength = P[ys, xs] / bg
    order = np.argsort(-strength)
    print(f"\n  {len(ys)} significant peaks (>8x background), top 10:")
    for i in order[:10]:
        rr = r[ys[i], xs[i]] * dq
        print(
            f"    ({xs[i]-cx:+3d},{ys[i]-cy:+3d})  |q|={rr:.4f} A^-1  "
            f"d={1/rr if rr>0 else float('inf'):.2f} A  strength={strength[i]:.1f}x"
        )
    lattice = [
        i for i in range(len(ys))
        if abs(r[ys[i], xs[i]] * dq - 1 / 3.28) / (1 / 3.28) < 0.18
        or abs(r[ys[i], xs[i]] * dq - 1 / 1.894) / (1 / 1.894) < 0.18
    ]
    print(f"\n  peaks at the WSe2 3.28 A or 1.89 A spacing: {len(lattice)} / {len(ys)}")
else:
    print("\n  NO significant peaks above 8x background")

print(f"\n  peak/background ratio of spectrum: {P[ring].max()/bg:.1f}x")

# ---- per-slice lattice coherence ----------------------------------------- #
print("\n=== per-slice lattice coherence ===")
target = 1 / 3.28
for z in range(min(4, phase.shape[0])):
    s = phase[z] - phase[z].mean()
    Fs = np.abs(np.fft.fftshift(np.fft.fft2(s * win))) ** 2
    band = (np.abs(r * dq - target) < 0.12 * target)
    print(f"  slice {z:02d}: (100) band power / total = {Fs[band].sum()/Fs.sum():.4f}")
