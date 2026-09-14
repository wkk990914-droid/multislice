"""End-to-end orchestration of the multislice ptychography workflow.

Stages run in the order given by the configuration; each stage can either build
a fresh model, clone a previous one, or initialise a DIP network from a
pixelated result.

Typical use::

    from ptycho import MultislicePtychoPipeline, PipelineConfig

    cfg = PipelineConfig.from_yaml("configs/multislice_default.yaml")
    pipeline = MultislicePtychoPipeline(cfg)
    results = pipeline.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from quantem.core import config as qconfig
from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging import Ptychography, PtychographyDatasetRaster

from .config import PipelineConfig, StageConfig
from .io import (
    build_dataset4dstem,
    calibrate_diffraction_sampling,
    load_scan,
    save_array,
)
from .models import PixelatedModels, build_pixelated_models
from .stages import build_stage, run_stage
from .viz import (
    FigureSink,
    build_virtual_images,
    configure_matplotlib,
    plot_linescan,
    show_depth_profile,
    show_mean_diffraction,
    show_virtual_images,
)

logger = logging.getLogger(__name__)


class MultislicePtychoPipeline:
    """Load the data, run the configured reconstruction stages, save artefacts."""

    def __init__(self, config: PipelineConfig) -> None:
        cfg = config
        cfg.validate()
        self.config = cfg

        self.output_dir = Path(cfg.output.dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sink = FigureSink(
            self.output_dir, show=cfg.output.show, enabled=cfg.output.save_figures
        )

        self.dset: Dataset4dstem | None = None
        self.pdset: PtychographyDatasetRaster | None = None
        self.pixelated: PixelatedModels | None = None
        self.results: dict[str, Ptychography] = {}
        self.analysis: dict[str, dict[str, Any]] = {}

    # -- setup ------------------------------------------------------------
    def setup_device(self) -> None:
        configure_matplotlib(self.config.output.show)
        qconfig.set_device(self.config.device.gpu_id)
        logger.info("Using device %s", qconfig.get("device"))

    # -- data -------------------------------------------------------------
    def load_data(self) -> tuple[Dataset4dstem, PtychographyDatasetRaster]:
        """Read the raw file, calibrate diffraction sampling and preprocess the scan."""
        cfg = self.config
        array = load_scan(cfg.data)
        dset = build_dataset4dstem(array, cfg.data, cfg.experiment)
        dset.get_dp_mean()
        show_mean_diffraction(dset, self.sink)

        center_y, center_x, radius = calibrate_diffraction_sampling(
            dset, cfg.experiment, cfg.data
        )
        logger.info("Probe disc: center=(%.2f, %.2f), R=%.2f px", center_y, center_x, radius)
        logger.info("Dataset:\n%s", dset)

        build_virtual_images(
            dset, (center_y, center_x), radius, cfg.preprocess, self.sink
        )
        show_virtual_images(dset, self.sink)

        pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
        pdset.preprocess(
            com_fit_function=cfg.preprocess.com_fit_function,
            plot_rotation=cfg.preprocess.plot_rotation,
            plot_com=cfg.preprocess.plot_com,
            probe_energy=cfg.experiment.probe_energy,
            force_com_rotation=cfg.preprocess.force_com_rotation,
        )

        self.dset, self.pdset = dset, pdset
        if self.pixelated is None:
            self.pixelated = build_pixelated_models(cfg.model, cfg.experiment)
        return dset, pdset

    # -- execution --------------------------------------------------------
    def run(self, stage_names: list[str] | None = None) -> dict[str, Ptychography]:
        """Run every (selected) stage in order."""
        self.setup_device()
        if self.pdset is None:
            self.load_data()
        assert self.pdset is not None and self.pixelated is not None

        stages = self.config.stages
        if stage_names is not None:
            selected = set(stage_names)
            stages = [s for s in stages if s.name in selected]
            missing = selected - {s.name for s in stages}
            if missing:
                raise KeyError(f"Unknown stage(s): {sorted(missing)}")

        for stage in stages:
            self.run_one(stage)
        return self.results

    def run_one(self, stage: StageConfig) -> Ptychography:
        """Build (or clone) the model for a stage, reconstruct and analyse it."""
        assert self.pdset is not None and self.pixelated is not None

        ptycho = build_stage(stage, self.config, self.pdset, self.results, self.pixelated)
        ptycho = run_stage(ptycho, stage, self.config)

        if self.config.output.save_figures or self.config.output.show:
            with self.sink.capture(f"{stage.name}_reconstruct"):
                ptycho.visualize()

        self.results[stage.name] = ptycho
        self._analyse(stage, ptycho)
        self._save(stage, ptycho)
        return ptycho

    # -- analysis ---------------------------------------------------------
    def _analyse(self, stage: StageConfig, ptycho: Ptychography) -> None:
        analysis = self.config.analysis
        payload: dict[str, Any] = {}

        if stage.show_obj:
            with self.sink.capture(f"{stage.name}_object"):
                ptycho.show_obj(axsize=(10, 10))

        if stage.linescan:
            length, profile = plot_linescan(
                ptycho, analysis, self.sink, name=f"{stage.name}_linescan"
            )
            payload["linescan_length"] = np.asarray(length)
            payload["linescan_profile"] = np.asarray(profile)

        if stage.depth_profile:
            profile = payload.get("linescan_profile")
            if profile is None:
                _, profile = plot_linescan(
                    ptycho, analysis, self.sink, name=f"{stage.name}_linescan"
                )
                payload["linescan_profile"] = np.asarray(profile)
            resampled = show_depth_profile(
                ptycho, profile, analysis, self.sink, name=f"{stage.name}_depth_profile"
            )
            payload["depth_profile"] = np.asarray(resampled)

        self.analysis[stage.name] = payload

    def _save(self, stage: StageConfig, ptycho: Ptychography) -> None:
        if not self.config.output.save_arrays:
            return
        stage_dir = self.output_dir / stage.name
        try:
            save_array(ptycho.obj_cropped, stage_dir / "obj_cropped.npy")
        except Exception as exc:  # pragma: no cover - model dependent
            logger.warning("Could not save object for stage %s: %s", stage.name, exc)

        for key, value in self.analysis.get(stage.name, {}).items():
            try:
                save_array(value, stage_dir / f"{key}.npy")
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not save %s for stage %s: %s", key, stage.name, exc)
