"""Visualisation helpers.

All plotting is funnelled through :class:`FigureSink`, which optionally writes
the figures produced by ``quantem``'s ``show=True`` calls to disk and closes
them when running headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import matplotlib

from .config import AnalysisConfig


def configure_matplotlib(show: bool) -> None:
    """Select a non-interactive backend when figures should not be displayed."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (import after backend selection)


def _plt():
    import matplotlib.pyplot as plt

    return plt


class FigureSink:
    """Save (and optionally close) every figure created inside a ``with`` block.

    Example
    -------
    >>> with sink("01_mean_dp"):
    ...     dset.dp_mean.show(cbar=True)
    """

    def __init__(self, output_dir: Path, show: bool = True, enabled: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.show = show
        self.enabled = enabled
        self._seen: set[int] = set()

    def __call__(self, name: str) -> "FigureSink":
        """Return a fresh sink sharing this configuration."""
        return FigureSink(self.output_dir, show=self.show, enabled=self.enabled)

    def __enter__(self) -> "FigureSink":
        plt = _plt()
        self._seen = set(plt.get_fignums())
        return self

    def __exit__(self, *exc_info: Any) -> None:
        plt = _plt()
        new_figures = sorted(set(plt.get_fignums()) - self._seen)
        prefix = getattr(self, "_current_name", "figure")
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for index, num in enumerate(new_figures):
                suffix = f"_{index}" if len(new_figures) > 1 else ""
                plt.figure(num).savefig(
                    self.output_dir / f"{prefix}{suffix}.png", dpi=150, bbox_inches="tight"
                )
        if not self.show:
            for num in new_figures:
                plt.close(num)

    def capture(self, name: str) -> "FigureSink":
        """Name the next captured figure group."""
        self._current_name = name
        return self


def show_mean_diffraction(dset: Any, sink: FigureSink, name: str = "00_dp_mean") -> None:
    with sink.capture(name):
        dset.dp_mean.show(upper_quantile=0.97, cbar=True)


def show_virtual_images(dset: Any, sink: FigureSink, name: str = "03_virtual_images") -> None:
    with sink.capture(name):
        dset.show_virtual_images(cmap="viridis")


def build_virtual_images(
    dset: Any,
    probe_center: tuple[float, float],
    probe_radius: float,
    pre_cfg: Any,
    sink: FigureSink,
) -> None:
    """Add the dark-field / bright-field virtual images used for sanity checks."""
    with sink.capture("02_virtual_image_df"):
        dset.get_virtual_image(
            mode="annular",
            geometry=(
                (probe_center[0], probe_center[1]),
                (probe_radius + pre_cfg.df_inner_pad, probe_radius * pre_cfg.df_outer_factor),
            ),
            name="DF",
            show=True,
        )
    with sink.capture("02_virtual_image_bf"):
        dset.get_virtual_image(
            mode="circle",
            geometry=(
                (probe_center[0], probe_center[1]),
                probe_radius + pre_cfg.bf_radius_pad,
            ),
            name="BF",
            show=True,
        )


def plot_linescan(
    ptycho: Any,
    analysis: AnalysisConfig,
    sink: FigureSink,
    name: str,
) -> tuple[Any, Any]:
    """Linescan through the reconstructed object, returns ``(length, profile)``."""
    from quantem.core.visualization import linescan

    with sink.capture(name):
        length, profile = linescan(
            ptycho.obj_cropped,
            center=(int(analysis.center[0]), int(analysis.center[1])),
            phi=analysis.phi,
            # ``quantem.linescan`` slices with ``line_len`` and needs an integer
            # (plus an integer ``linewidth`` for ``profile_line``).
            line_len=int(round(analysis.line_len)),
            show=True,
            linewidth=int(round(analysis.linewidth)),
            cmap=analysis.cmap,
        )
    return length, profile


def show_depth_profile(
    ptycho: Any,
    profile: Any,
    analysis: AnalysisConfig,
    sink: FigureSink,
    name: str = "depth_profile",
) -> Any:
    """Resample a z-x linescan profile onto the physical depth axis and show it."""
    import skimage.transform
    from quantem.core.visualization import show_2d

    resample_factor_z = float(ptycho.slice_thicknesses[0]) / float(ptycho.sampling[0])
    profile = to_array(profile)
    resampled = skimage.transform.resize(
        profile,
        (profile.shape[0] * resample_factor_z, profile.shape[1]),
        order=1,
        mode="edge",
    )
    with sink.capture(name):
        show_2d(
            resampled,
            cmap=analysis.cmap,
            axsize=(5, 5),
            upper_quantile=0.98,
            lower_quantile=0.5,
            scalebar=[{"sampling": ptycho.sampling[0], "units": r"$\mathrm{\AA}$"}],
        )
    return resampled


def to_array(array: Any):
    from .io import to_numpy

    return to_numpy(array)


def iter_figures() -> Iterator[int]:
    return iter(_plt().get_fignums())
