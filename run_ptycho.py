#!/usr/bin/env python
"""Command line entry point for the multislice ptychography workflow.

Examples
--------
Run the full pipeline::

    python run_ptycho.py --config configs/multislice_default.yaml

Only run the pixelated stages, headless, custom output directory::

    python run_ptycho.py --stages pixelated_baseline pixelated_baseline_extra \\
        pixelated_free --no-show --output-dir runs/pixelated

Dump an effective configuration resolved from the defaults::

    python run_ptycho.py --print-config
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ptycho import MultislicePtychoPipeline, PipelineConfig

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "multislice_default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multislice ptychography reconstruction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG, help="YAML configuration file"
    )
    parser.add_argument(
        "-s",
        "--stages",
        nargs="*",
        default=None,
        help="Only run these stage names (default: all stages in the config)",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Override output.dir")
    parser.add_argument(
        "--device",
        default=None,
        choices=["gpu", "cpu"],
        help="Override the reconstruction device for every stage",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=None, help="Override device.gpu_id"
    )
    parser.add_argument(
        "--num-iters", type=int, default=None, help="Override num_iters for every selected stage"
    )
    parser.add_argument("--show", dest="show", action="store_true", help="Open interactive figures")
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Headless mode: save figures instead of displaying them",
    )
    parser.set_defaults(show=None)
    parser.add_argument(
        "--print-config", action="store_true", help="Print the resolved configuration and exit"
    )
    parser.add_argument(
        "--save-config", type=Path, default=None, help="Write the resolved configuration and exit"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity"
    )
    return parser


def apply_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    if args.output_dir is not None:
        cfg.output.dir = str(args.output_dir)
    if args.gpu_id is not None:
        cfg.device.gpu_id = args.gpu_id
    if args.device is not None:
        cfg.device.default_device = args.device
        for stage in cfg.stages:
            stage.device = args.device
    if args.show is not None:
        cfg.output.show = args.show
    if args.num_iters is not None:
        for stage in cfg.stages:
            stage.num_iters = args.num_iters
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )

    cfg = PipelineConfig.from_yaml(args.config)
    cfg = apply_overrides(cfg, args)
    cfg.validate()

    if args.print_config:
        import yaml

        yaml.safe_dump(cfg.to_dict(), sys.stdout, sort_keys=False)
        return 0
    if args.save_config is not None:
        cfg.to_yaml(args.save_config)
        print(f"Wrote resolved configuration to {args.save_config}")
        return 0

    pipeline = MultislicePtychoPipeline(cfg)
    pipeline.run(stage_names=args.stages)
    print(f"Finished. Results written to {pipeline.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
