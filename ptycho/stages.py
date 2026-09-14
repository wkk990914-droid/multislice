"""Individual reconstruction stages (pixelated / deep image prior)."""

from __future__ import annotations

import logging
from typing import Any

from quantem.diffractive_imaging import ObjectDIP, Ptychography

from .compat import object_constraints
from .config import ConstraintConfig, PipelineConfig, StageConfig
from .models import (
    PixelatedModels,
    build_dip_models,
    build_optimizer_params,
    build_pixelated_models,
    build_scheduler_params,
)

logger = logging.getLogger(__name__)


def build_constraints(
    constraints: ConstraintConfig | None,
) -> dict[str, Any] | None:
    """Translate a :class:`ConstraintConfig` into the ``quantem`` kwargs."""
    if constraints is None:
        return None
    kwargs: dict[str, Any] = {
        "identical_slices": constraints.identical_slices,
        "fix_potential_baseline": constraints.fix_potential_baseline,
        "fix_potential_baseline_factor": constraints.fix_potential_baseline_factor,
    }
    if constraints.positivity_mode is not None:
        kwargs["positivity_mode"] = constraints.positivity_mode
    if constraints.tv_weight_z is not None:
        kwargs["tv_weight_z"] = constraints.tv_weight_z
    if constraints.surface_zero_weight is not None:
        kwargs["surface_zero_weight"] = constraints.surface_zero_weight
    return {"object": object_constraints(**kwargs)}


def build_stage(
    stage: StageConfig,
    cfg: PipelineConfig,
    pdset: Any,
    registry: dict[str, Ptychography],
    pixelated: PixelatedModels,
) -> Ptychography:
    """Create the :class:`Ptychography` object a stage will optimise.

    ``init_from`` resolution:

    * ``None``                        -> fresh pixelated model
    * source is pixelated             -> ``clone()`` (pixelated) or DIP init (dip kind)
    * source is DIP and kind is "dip" -> ``clone()``
    """
    source = registry.get(stage.init_from) if stage.init_from else None

    if stage.kind == "dip" and not (source is not None and isinstance(source.obj_model, ObjectDIP)):
        # Initialise the DIP networks from the (converged) pixelated object.
        if source is None:
            raise ValueError(
                f"Stage {stage.name!r} is a DIP stage and must set `init_from` "
                "to a pixelated stage."
            )
        obj_model_dip, probe_model_dip = build_dip_models(
            cfg.dip, cfg.model, source.obj_model, source.probe_model
        )
        build_device = stage.build_device or cfg.device.build_device
        ptycho = Ptychography.from_models(
            dset=pdset,
            obj_model=obj_model_dip,
            probe_model=probe_model_dip,
            detector_model=pixelated.detector,
            device=build_device,
        )
        ptycho.preprocess(
            obj_padding_px=tuple(cfg.preprocess.obj_padding_px),
            **({"batch_size": stage.preprocess_batch_size} if stage.preprocess_batch_size else {}),
        )
        return ptycho

    if source is not None:
        return source.clone()

    if stage.kind != "pixelated":  # pragma: no cover - guarded by validate()
        raise ValueError(f"Stage {stage.name!r}: cannot build kind {stage.kind!r} from scratch")

    build_device = stage.build_device or cfg.device.build_device
    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=pixelated.obj,
        probe_model=pixelated.probe,
        detector_model=pixelated.detector,
        device=build_device,
    )
    ptycho.preprocess(
        obj_padding_px=tuple(cfg.preprocess.obj_padding_px),
        **({"batch_size": stage.preprocess_batch_size} if stage.preprocess_batch_size else {}),
    )
    return ptycho


def run_stage(ptycho: Ptychography, stage: StageConfig, cfg: PipelineConfig) -> Ptychography:
    """Run a single ``reconstruct`` call with the stage's optimizer settings."""
    kwargs: dict[str, Any] = {
        "num_iters": stage.num_iters,
        "reset": stage.reset,
        "device": stage.device or cfg.device.default_device,
    }
    if stage.batch_size is not None:
        kwargs["batch_size"] = stage.batch_size
    if stage.store_snapshots_every is not None:
        kwargs["store_snapshots_every"] = stage.store_snapshots_every
    if stage.optimizer is not None:
        kwargs["optimizer_params"] = build_optimizer_params(stage.optimizer)
        kwargs["scheduler_params"] = build_scheduler_params(stage.optimizer)
    constraints = build_constraints(stage.constraints)
    if constraints is not None:
        kwargs["constraints"] = constraints

    logger.info("Stage %s: %d iterations on %s", stage.name, stage.num_iters, kwargs["device"])
    return ptycho.reconstruct(**kwargs)
