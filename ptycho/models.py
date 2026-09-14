"""Forward-model construction: pixelated models and their DIP counterparts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from quantem.core import config as qconfig
from quantem.core.ml import CNN2d, OptimizerParams, SchedulerParams
from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectDIP,
    ObjectPixelated,
    ProbeDIP,
    ProbePixelated,
)

from .config import DipConfig, ExperimentConfig, ModelConfig, OptimConfig


@dataclass
class PixelatedModels:
    """Container for the three pixelated forward-model components."""

    obj: ObjectPixelated
    probe: ProbePixelated
    detector: DetectorPixelated


def build_pixelated_models(
    model_cfg: ModelConfig, exp_cfg: ExperimentConfig
) -> PixelatedModels:
    """Uniform potential object + parameterised probe + pixelated detector."""
    obj_model = ObjectPixelated.from_uniform(
        num_slices=model_cfg.num_slices,
        slice_thicknesses=model_cfg.slice_thickness,
        obj_type=model_cfg.obj_type,
    )
    probe_model = ProbePixelated.from_params(
        num_probes=model_cfg.num_probes,
        probe_params=exp_cfg.probe_params,
    )
    detector_model = DetectorPixelated()
    return PixelatedModels(obj=obj_model, probe=probe_model, detector=detector_model)


def build_dip_models(
    dip_cfg: DipConfig,
    model_cfg: ModelConfig,
    pixelated_obj: ObjectPixelated,
    pixelated_probe: ProbePixelated,
    device: str | None = None,
) -> tuple[ObjectDIP, ProbeDIP]:
    """Wrap the pixelated models in CNNs and pretrain them to match.

    Pretraining gives the subsequent free-slice optimisation a stable starting
    point instead of a random network output.
    """
    device = device or qconfig.get("device")

    cnn_obj = CNN2d(
        in_channels=model_cfg.num_slices,
        final_activation=dip_cfg.obj_final_activation,
    )
    obj_model_dip = ObjectDIP.from_pixelated(
        model=cnn_obj,
        pixelated=pixelated_obj,
        device=device,
    )

    cnn_probe = CNN2d(
        in_channels=model_cfg.num_probes,
        start_filters=dip_cfg.probe_start_filters,
        dtype=torch.complex64,
    )
    probe_model_dip = ProbeDIP.from_pixelated(
        model=cnn_probe,
        pixelated=pixelated_probe,
        device=device,
    )

    obj_model_dip.pretrain(
        num_iters=dip_cfg.obj_pretrain_iters,
        optimizer_params=OptimizerParams.Adam(lr=dip_cfg.pretrain_lr),
        scheduler_params=SchedulerParams.Plateau(),
    )
    probe_model_dip.pretrain(
        num_iters=dip_cfg.probe_pretrain_iters,
        optimizer_params=OptimizerParams.Adam(lr=dip_cfg.pretrain_lr),
        scheduler_params=SchedulerParams.Plateau(),
    )
    return obj_model_dip, probe_model_dip


def build_optimizer_params(opt_cfg: OptimConfig) -> dict[str, Any]:
    """Build the ``{"object": ..., "probe": ...}`` optimizer mapping."""
    factory = getattr(OptimizerParams, opt_cfg.optimizer)
    return {
        "object": factory(lr=opt_cfg.object_lr),
        "probe": factory(lr=opt_cfg.probe_lr),
    }


def build_scheduler_params(opt_cfg: OptimConfig) -> dict[str, Any]:
    """Build the ``{"object": ..., "probe": ...}`` scheduler mapping."""
    factory = getattr(SchedulerParams, opt_cfg.scheduler)
    return {"object": factory(), "probe": factory()}
