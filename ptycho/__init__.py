"""Refactored, scriptable version of the multislice ptychography tutorial.

Modules
-------
config     dataclass + YAML configuration for the whole workflow
io         EMPAD raw loading, diffraction calibration, array persistence
viz        figure handling (interactive or saved to disk)
models     pixelated / deep-image-prior forward models
stages     per-stage build + reconstruct logic
pipeline   end-to-end orchestration
"""

from .config import (
    AnalysisConfig,
    ConstraintConfig,
    DataConfig,
    DeviceConfig,
    DipConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessConfig,
    StageConfig,
)

__all__ = [
    "AnalysisConfig",
    "ConstraintConfig",
    "DataConfig",
    "DeviceConfig",
    "DipConfig",
    "ExperimentConfig",
    "ModelConfig",
    "OptimConfig",
    "OutputConfig",
    "PipelineConfig",
    "PreprocessConfig",
    "StageConfig",
    "MultislicePtychoPipeline",
]


def __getattr__(name: str):
    if name == "MultislicePtychoPipeline":
        from .pipeline import MultislicePtychoPipeline

        return MultislicePtychoPipeline
    raise AttributeError(name)
