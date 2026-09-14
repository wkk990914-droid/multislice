"""Single-stage worker: loads data, restores a checkpoint, reconstructs, saves.

Spawned once per DAG job by ``run_atomic_phase_test.py`` with
``CUDA_VISIBLE_DEVICES`` already pinned to one physical GPU, so the worker
always addresses its card as device 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantem.core import config as qconfig
from quantem.diffractive_imaging import ObjectDIP, Ptychography

from .analysis import check_finite, to_numpy
from .config import PipelineConfig, StageConfig
from .io import (
    build_dataset4dstem,
    calibrate_diffraction_sampling,
    load_scan,
)
from .models import build_pixelated_models
from .stages import build_stage, run_stage

logger = logging.getLogger("ptycho.worker")


def checkpoint_dir(output_dir: Path, stage_name: str) -> Path:
    return output_dir / "checkpoints" / stage_name


def save_checkpoint(
    ptycho: Ptychography, output_dir: Path, stage_name: str, kind: str
) -> Path:
    target = checkpoint_dir(output_dir, stage_name)
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "ptycho.zip"
    ptycho.save(archive, mode="o", verbose=False)
    (target / "meta.json").write_text(
        json.dumps({"stage": stage_name, "kind": kind}), encoding="utf-8"
    )
    return archive


def load_checkpoint(
    output_dir: Path, stage_name: str, dset: Any = None, device: str | None = None
) -> Ptychography | None:
    """Restore a stage checkpoint.

    ``Ptychography.save`` does **not** persist the dataset, so ``dset`` must be
    re-supplied; every worker prepares it identically from the same config.
    """
    target = checkpoint_dir(output_dir, stage_name)
    archive = target / "ptycho.zip"
    if not archive.exists():
        return None
    return Ptychography.from_file(archive, dset=dset, device=device, verbose=False)


def checkpoint_kind(output_dir: Path, stage_name: str) -> str | None:
    meta = checkpoint_dir(output_dir, stage_name) / "meta.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text(encoding="utf-8")).get("kind")


def peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / 1024**2
    return 0.0


def prepare_dataset(cfg: PipelineConfig):
    """Load + calibrate + preprocess, returning ``(Dataset4dstem, pdset)``."""

    from quantem.diffractive_imaging import PtychographyDatasetRaster

    array = load_scan(cfg.data)
    dset = build_dataset4dstem(array, cfg.data, cfg.experiment)
    dset.get_dp_mean()
    calibrate_diffraction_sampling(dset, cfg.experiment, cfg.data)

    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(
        com_fit_function=cfg.preprocess.com_fit_function,
        plot_rotation=cfg.preprocess.plot_rotation,
        plot_com=cfg.preprocess.plot_com,
        probe_energy=cfg.experiment.probe_energy,
        force_com_rotation=cfg.preprocess.force_com_rotation,
    )
    return dset, pdset


def build_for_stage(
    stage: StageConfig,
    cfg: PipelineConfig,
    pdset: Any,
    output_dir: Path,
    device: str,
    continue_from_self: bool = False,
) -> Ptychography:
    """Restore from a checkpoint when possible, otherwise build fresh.

    Mirrors ``ptycho.stages.build_stage``: a source that is already the right
    model family is cloned, while a DIP stage built on a pixelated source is
    re-initialised from that source's trained object / probe networks.

    With ``continue_from_self`` the stage's *own* checkpoint is resumed instead
    of its ``init_from`` source, which is what a cumulative iteration extension
    needs; falling back to the normal path when no own checkpoint exists.
    """
    pixelated_models = build_pixelated_models(cfg.model, cfg.experiment)

    source: Ptychography | None = None
    if continue_from_self:
        source = load_checkpoint(output_dir, stage.name, dset=pdset, device=device)
        if source is not None:
            return source
    if stage.init_from and source is None:
        source = load_checkpoint(output_dir, stage.init_from, dset=pdset, device=device)

    registry: dict[str, Ptychography] = {}
    if source is not None:
        registry[stage.init_from] = source

    return build_stage(stage, cfg, pdset, registry, pixelated_models)


def run_one_stage(
    config_path: Path,
    stage_name: str,
    output_dir: Path,
    gpu_label: str,
    continue_from_self: bool = False,
) -> dict[str, Any]:
    started = time.time()
    seed = os.environ.get("PYREX_SEED")
    if seed is not None:
        seed_i = int(seed)
        np.random.seed(seed_i)
        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)
    cfg = PipelineConfig.from_yaml(config_path)
    stage = cfg.stage(stage_name)

    # CUDA_VISIBLE_DEVICES already pins the physical card, so the worker always
    # sees its GPU as index 0.  Fall back to CPU when no CUDA is present.
    if torch.cuda.is_available():
        qconfig.set_device(0)
        logger.info("stage=%s gpu_label=%s device=%s", stage_name, gpu_label, qconfig.device())
    else:
        logger.info("stage=%s gpu_label=%s device=cpu (no cuda)", stage_name, gpu_label)

    _dset, pdset = prepare_dataset(cfg)
    # Build / restore on the CPU, then let run_stage move the model to the GPU.
    ptycho = build_for_stage(
        stage, cfg, pdset, output_dir, cfg.device.build_device,
        continue_from_self=continue_from_self,
    )

    # A checkpoint carries its own _iter_losses, so only the entries appended by
    # *this* call count as this call's loss history.  When extending a stage from
    # its own checkpoint the history is cumulative, so keep everything.
    prior = len(np.asarray(ptycho.iter_losses))
    ptycho = run_stage(ptycho, stage, cfg)

    all_losses = np.asarray(ptycho.iter_losses, dtype=np.float64)
    if continue_from_self:
        losses = all_losses
    else:
        losses = all_losses[prior:] if prior <= all_losses.size else all_losses

    obj = to_numpy(ptycho.obj_cropped)
    probe = to_numpy(ptycho.probe_model.probe)

    report = check_finite(f"{stage_name}.object", obj)
    report.update(check_finite(f"{stage_name}.probe", probe))
    if losses.size:
        report.update(check_finite(f"{stage_name}.loss", losses))

    stage_dir = output_dir / "stages" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.save(stage_dir / "obj_cropped.npy", obj)
    np.save(stage_dir / "probe.npy", probe)
    np.save(stage_dir / "loss_history.npy", losses)
    # The object grid is resampled during preprocessing (descanning + rotation
    # interpolation), so record the *actual* pixel size rather than the raw
    # scan step; the FFT analysis depends on it for correct d-spacings.
    sampling = float(np.asarray(ptycho.sampling).ravel()[0])
    save_checkpoint(ptycho, output_dir, stage_name, stage.kind)

    metrics: dict[str, Any] = {
        "stage": stage_name,
        "kind": stage.kind,
        "num_iters_requested": stage.num_iters,
        "iterations_recorded": int(losses.size),
        "initial_loss": float(losses[0]) if losses.size else None,
        "final_loss": float(losses[-1]) if losses.size else None,
        "min_loss": float(losses.min()) if losses.size else None,
        "loss_history": [float(v) for v in losses],
        "object_shape": list(obj.shape),
        "object_dtype": str(obj.dtype),
        "object_sampling_A": sampling,
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "runtime_s": round(time.time() - started, 2),
        "gpu_label": gpu_label,
        "finite": True,
        "finite_report": report,
    }
    (stage_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"STAGE_METRICS {json.dumps(metrics)}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run one ptychography stage")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-label", default="unknown")
    parser.add_argument(
        "--continue-from-self",
        action="store_true",
        help="resume this stage's own checkpoint (cumulative iterations)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO - 10 * min(args.verbose, 2),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
    import matplotlib

    matplotlib.use("Agg")

    run_one_stage(
        args.config, args.stage, args.output_dir, args.gpu_label,
        continue_from_self=args.continue_from_self,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
