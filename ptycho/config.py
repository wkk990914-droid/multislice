"""Configuration objects for the multislice ptychography workflow.

Every default value mirrors the original tutorial notebook
(``ptycho_iter_05_multislice.ipynb``) so the refactored code reproduces the
notebook results one-to-one.

The configuration tree is plain dataclasses and can be round-tripped to YAML::

    cfg = PipelineConfig.from_yaml("configs/multislice_default.yaml")
    cfg.to_yaml("configs/my_run.yaml")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


def _unwrap_optional(annotation: Any) -> Any:
    """Return the inner type of ``Optional[X]`` / ``X | None``, else the type itself."""
    args = get_args(annotation)
    if args and type(None) in args:
        return args[0]
    return annotation


def _from_dict(cls: type, data: dict[str, Any] | None):
    """Recursively build a dataclass instance from a (nested) dictionary."""
    if data is None:
        return None
    if is_dataclass(data):
        return data

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in dict(data).items():
        if name not in {f.name for f in fields(cls)}:
            raise KeyError(f"Unknown config key {name!r} for {cls.__name__}")
        target = _unwrap_optional(hints.get(name, Any))
        origin = get_origin(target)

        if is_dataclass(target) and isinstance(value, dict):
            kwargs[name] = _from_dict(target, value)
        elif origin is list:
            item = get_args(target)[0] if get_args(target) else Any
            if is_dataclass(item):
                kwargs[name] = [_from_dict(item, v) for v in value]
            else:
                kwargs[name] = list(value)
        elif origin is tuple and isinstance(value, (list, tuple)):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


@dataclass
class DeviceConfig:
    """Compute settings shared by every stage."""

    gpu_id: int = 0
    default_device: str = "gpu"
    # Device used when constructing a fresh Ptychography object.
    build_device: str = "cpu"


@dataclass
class DataConfig:
    """Description of the input 4D-STEM file and how to fold it into a dataset.

    Two formats are supported, selected by the file suffix:

    * ``.raw``   - EMPAD-style binary with inter-frame gaps (the tutorial data).
    * ``.h5``    - HDF5 / NeXus 4D-STEM (e.g. PYREX simulations). ``group`` and
      ``dataset`` narrow the search; ``autodetect_axes`` permutes axes using the
      ``Axis{i}_quantity`` attributes.
    """

    raw_path: str = "~/data/example_data/Panel_g-h_Themis/scan_x128_y128.raw"
    name: str = "tBL_WSe2"
    scan_shape: tuple[int, int] = (128, 128)
    dp_shape: tuple[int, int] = (128, 128)
    dtype: str = "float32"
    gap_bytes: int = 1024
    offset_bytes: int = 0
    transpose_dp: bool = True
    flip_dp_x: bool = False
    flip_dp_y: bool = False
    transpose_scan: bool = False
    flip_scan_x: bool = False
    flip_scan_y: bool = False
    # HDF5 only -----------------------------------------------------------
    group: str | None = None
    dataset: str | None = None
    autodetect_axes: bool = True
    # Reciprocal sampling in mrad/pixel; skips the probe-disc fit when set.
    dp_sampling_mrad: float | None = None
    # Optional sub-region selection, as [start, stop] pixel pairs. Useful for
    # quick smoke tests on large 4D datasets.
    scan_crop: tuple[int, int, int, int] | None = None  # (y0, y1, x0, x1)
    dp_crop: tuple[int, int, int, int] | None = None  # (qy0, qy1, qx0, qx1)

    @property
    def num_frames(self) -> int:
        ny, nx = self.scan_shape
        return ny * nx

    def resolved_path(self) -> Path:
        return Path(self.raw_path).expanduser()


@dataclass
class ExperimentConfig:
    """Microscope / experimental constants (explicitly set for reproducibility)."""

    probe_energy: float = 80e3
    probe_semiangle: float = 24.9
    probe_defocus: float = 0.0
    scan_step_size: float = 0.4290  # Angstrom

    @property
    def probe_params(self) -> dict[str, Any]:
        return {
            "energy": self.probe_energy,
            "defocus": self.probe_defocus,
            "semiangle_cutoff": self.probe_semiangle,
        }


@dataclass
class ModelConfig:
    """Forward-model shape: number of probes, slices and slice thickness."""

    num_probes: int = 6
    num_slices: int = 16
    slice_thickness: float = 1.0
    obj_type: str = "potential"


@dataclass
class PreprocessConfig:
    """Scan preprocessing (CoM / rotation) and object padding."""

    com_fit_function: str = "constant"
    force_com_rotation: float | None = 93.0
    plot_rotation: bool = True
    plot_com: bool = True
    obj_padding_px: tuple[int, int] = (32, 32)
    batch_size: int = 128
    # Virtual-imaging geometry, expressed relative to the fitted probe radius.
    bf_radius_pad: float = 2.0
    df_inner_pad: float = 10.0
    df_outer_factor: float = 5.0


@dataclass
class DipConfig:
    """Deep-image-prior network and pretraining settings."""

    obj_pretrain_iters: int = 1500
    probe_pretrain_iters: int = 100
    pretrain_lr: float = 5e-3
    obj_final_activation: str | None = "softplus"
    probe_start_filters: int = 32


@dataclass
class OptimConfig:
    """Optimizer / scheduler pair used by a reconstruction stage."""

    optimizer: str = "Adam"
    scheduler: str = "Plateau"
    object_lr: float = 1e-4
    probe_lr: float = 1e-3


@dataclass
class ConstraintConfig:
    """Object constraints forwarded to ``PtychoObjConstraintParams.Raster``."""

    identical_slices: bool = True
    fix_potential_baseline: bool = True
    fix_potential_baseline_factor: float = 1.0
    positivity_mode: str | None = None
    tv_weight_z: float | None = None
    surface_zero_weight: float | None = None


@dataclass
class StageConfig:
    """A single reconstruction stage.

    ``kind`` is one of ``"pixelated"`` or ``"dip"``.  ``init_from`` names a
    previously executed stage and its meaning depends on the source type:

    * ``None``                     -> build a fresh pixelated model
    * pixelated source             -> ``clone()`` it (pixelated) or use it as
                                      the DIP initialisation (dip)
    * DIP source + ``kind="dip"``  -> ``clone()`` the DIP model
    """

    name: str
    num_iters: int
    kind: str = "pixelated"
    init_from: str | None = None
    reset: bool = False
    batch_size: int | None = None
    device: str | None = None
    store_snapshots_every: int | None = None
    optimizer: OptimConfig | None = None
    constraints: ConstraintConfig | None = None
    # Device used when this stage has to build a new Ptychography object.
    build_device: str | None = None
    # Batch size passed to `preprocess` (omitted when null).
    preprocess_batch_size: int | None = None
    # Post-stage analysis switches.
    show_obj: bool = False
    linescan: bool = False
    depth_profile: bool = False


@dataclass
class AnalysisConfig:
    """Linescan / depth-profile geometry used for the final analysis."""

    center: tuple[int, int] = (205, 72)
    phi: float = 20.0
    line_len: float = 60.0
    linewidth: float = 10.0
    cmap: str = "viridis"


@dataclass
class OutputConfig:
    """Where to write figures and arrays, and whether to open interactive windows."""

    dir: str = "results"
    save_arrays: bool = True
    save_figures: bool = True
    show: bool = True


@dataclass
class PipelineConfig:
    """Top level configuration for :class:`~ptycho.pipeline.MultislicePtychoPipeline`."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    dip: DipConfig = field(default_factory=DipConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    stages: list[StageConfig] = field(default_factory=list)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return _from_dict(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(Path(path).expanduser(), "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    # -- helpers ----------------------------------------------------------
    def stage(self, name: str) -> StageConfig:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"No stage named {name!r}. Available: {[s.name for s in self.stages]}")

    def validate(self) -> None:
        """Fail early on typos / dangling ``init_from`` references."""
        if not self.stages:
            raise ValueError("PipelineConfig requires at least one stage.")
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate stage names: {names}")
        for stage in self.stages:
            if stage.kind not in {"pixelated", "dip"}:
                raise ValueError(f"Stage {stage.name!r}: unknown kind {stage.kind!r}")
            if stage.init_from is not None and stage.init_from not in names:
                raise ValueError(
                    f"Stage {stage.name!r} depends on unknown stage {stage.init_from!r}"
                )
            if stage.init_from is not None and names.index(stage.init_from) >= names.index(
                stage.name
            ):
                raise ValueError(
                    f"Stage {stage.name!r} must run after its source {stage.init_from!r}"
                )
