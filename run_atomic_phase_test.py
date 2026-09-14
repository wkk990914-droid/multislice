#!/usr/bin/env python
"""Atomic-phase reconstruction test with GPU 5/6/7 coarse-grained scheduling.

The reconstruction DAG is derived from ``init_from`` in the config: a stage may
only start once its checkpoint exists, so continuation semantics are preserved.
Stages that are mutually independent run concurrently on different cards.

    pixelated_baseline -> pixelated_baseline_extra -+-> pixelated_free
                                                    +-> dip -> dip_free -> dip_free_extra

Usage::

    python run_atomic_phase_test.py -c configs/pyrex_atomic_phase.yaml
    python run_atomic_phase_test.py --gpus 5,6,7 --analyze-only
    python run_atomic_phase_test.py --extend-final 100
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from ptycho.analysis import (
    check_finite,
    decompose_object,
    projected_amplitude,
    projected_phase_complex_product,
    projected_phase_unwrapped,
    robust_clim,
    save_loss_plot,
    save_montage,
    save_phase_fft,
    save_single,
    save_summary_figure,
    slice_depths,
    to_numpy,
    write_loss_csv,
)
from ptycho.config import PipelineConfig
from ptycho.gpu_scheduler import GpuPool, GpuScheduler, Job, write_stage_metrics
from ptycho.worker import checkpoint_dir

logger = logging.getLogger("ptycho.orchestrator")

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "pyrex_atomic_phase.yaml"
PERMITTED_GPUS = {5, 6, 7}


# --------------------------------------------------------------------------- #
# DAG
# --------------------------------------------------------------------------- #
def build_jobs(cfg: PipelineConfig, config_path: Path, output_dir: Path) -> list[Job]:
    """One subprocess job per stage, with dependencies from ``init_from``."""
    jobs: list[Job] = []
    for stage in cfg.stages:
        argv = [
            sys.executable,
            "-m",
            "ptycho.worker",
            "-c",
            str(config_path),
            "--stage",
            stage.name,
            "-o",
            str(output_dir),
        ]
        jobs.append(
            Job(
                name=stage.name,
                argv=argv,
                deps=[stage.init_from] if stage.init_from else [],
                log_path=output_dir / "logs" / f"{stage.name}.log",
            )
        )
    return jobs


def topology_report(cfg: PipelineConfig) -> str:
    """Explain how much parallelism the DAG actually allows."""
    dependents: dict[str, list[str]] = {}
    for stage in cfg.stages:
        if stage.init_from:
            dependents.setdefault(stage.init_from, []).append(stage.name)
    branches = {k: v for k, v in dependents.items() if len(v) > 1}
    width = max(len(v) for v in dependents.values()) if dependents else 1
    lines = ["stage dependency graph:"]
    for stage in cfg.stages:
        lines.append(f"  {stage.init_from or '(root)'} -> {stage.name}")
    lines.append(f"maximum concurrent stages in this DAG = {width}")
    if branches:
        lines.append(f"branch point(s): {branches}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyse_run(
    cfg: PipelineConfig,
    output_dir: Path,
    stage: str | None = None,
    prefix: str = "FINAL",
    write_shared_figures: bool = True,
) -> dict:
    """Produce per-slice, montage, projected-phase, FFT and convergence figures."""
    stage = stage or cfg.stages[-1].name
    stage_dir = output_dir / "stages" / stage
    obj = to_numpy(np.load(stage_dir / "obj_cropped.npy"))
    losses = np.load(stage_dir / "loss_history.npy")

    phase, amplitude, kind = decompose_object(obj)
    nz = phase.shape[0]
    thickness = float(cfg.model.slice_thickness)
    depths = slice_depths(nz, thickness)

    check_finite(f"final.{stage}.phase", phase)
    check_finite(f"final.{stage}.amplitude", amplitude)

    p_lo, p_hi = robust_clim(phase)
    a_lo, a_hi = robust_clim(amplitude) if kind == "complex" else (0.0, 1.0)

    titles = [f"Slice {z:02d}  z={depths[z]:.1f} A" for z in range(nz)]

    # ---- per-slice figures, shared colour scale -------------------------- #
    slices_dir = output_dir / "slices"
    if write_shared_figures:
        for z in range(nz):
            save_single(
                phase[z], slices_dir / f"slice_{z:02d}_phase.png", titles[z],
                cmap="viridis", vmin=p_lo, vmax=p_hi,
            )
            save_single(
                amplitude[z], slices_dir / f"slice_{z:02d}_amplitude.png", titles[z],
                cmap="magma", vmin=a_lo, vmax=a_hi,
            )

    montage_phase = save_montage(
        phase, output_dir / f"{prefix}_phase_all_slices_4x4.png", titles,
        cmap="viridis", vmin=p_lo, vmax=p_hi, cols=4,
        suptitle=f"phase - all {nz} slices ({stage}, {kind} object)",
    )
    montage_amp = save_montage(
        amplitude, output_dir / f"{prefix}_amplitude_all_slices_4x4.png", titles,
        cmap="magma", vmin=a_lo, vmax=a_hi, cols=4,
        suptitle=f"amplitude - all {nz} slices ({stage})",
    )

    # ---- projected phase ------------------------------------------------- #
    proj_unwrapped = projected_phase_unwrapped(phase)
    proj_product = projected_phase_complex_product(
        phase, amplitude if kind == "complex" else None
    )
    proj_amp = projected_amplitude(amplitude)
    q_lo, q_hi = robust_clim(proj_unwrapped)

    if write_shared_figures:
        save_single(
            proj_unwrapped, output_dir / "projected_phase_unwrapped.png",
            "projected phase (A: unwrap along z then sum)",
            cmap="twilight_shifted", vmin=q_lo, vmax=q_hi, dpi=300,
        )
        save_single(
            proj_product, output_dir / "projected_phase_complex_product.png",
            "projected phase (B: angle of prod_z exp(i*phase))",
            cmap="twilight_shifted", vmin=q_lo, vmax=q_hi, dpi=300,
        )
        save_single(
            proj_amp, output_dir / "projected_amplitude.png",
            "projected amplitude (prod_z |t|)", cmap="magma", dpi=300,
        )
    np.save(output_dir / f"{prefix}_projected_phase_unwrapped.npy", proj_unwrapped)

    # ---- FFT ------------------------------------------------------------- #
    # The object grid is resampled during preprocessing, so the true pixel size
    # comes from the worker's metrics, not from the raw scan step.
    sampling_a = _object_sampling(cfg, output_dir, stage)
    fft_info = save_phase_fft(
        proj_unwrapped,
        output_dir / ("phase_fft.png" if write_shared_figures else f"{prefix}_phase_fft.png"),
        sampling_a,
    )

    # ---- convergence ----------------------------------------------------- #
    histories = load_loss_histories(cfg, output_dir)
    if write_shared_figures:
        write_loss_csv(output_dir / "loss_history.csv", histories)
        loss_png = save_loss_plot(histories, output_dir / "loss_vs_iteration.png")
    else:
        loss_png = save_loss_plot(histories, output_dir / f"{prefix}_loss_vs_iteration.png")

    summary = None
    if write_shared_figures:
        try:
            summary = save_summary_figure(
                phase, titles, proj_unwrapped, fft_info, histories,
                output_dir / "FINAL_reconstruction_summary.png", p_lo, p_hi,
            )
        except Exception as exc:  # composite is best-effort
            logger.warning("summary figure failed: %s", exc)

    return {
        "stage": stage,
        "object_kind": kind,
        "object_shape": list(obj.shape),
        "phase_shape": list(phase.shape),
        "phase_min": float(phase.min()),
        "phase_max": float(phase.max()),
        "phase_std": float(phase.std()),
        "amplitude_min": float(amplitude.min()),
        "amplitude_max": float(amplitude.max()),
        "projected_phase_min": float(proj_unwrapped.min()),
        "projected_phase_max": float(proj_unwrapped.max()),
        "projected_phase_std": float(proj_unwrapped.std()),
        "projected_phase_B_std": float(proj_product.std()),
        "projected_phase_AB_max_diff": float(
            np.abs(proj_unwrapped - proj_product).max()
        ),
        "fft": {k: v for k, v in fft_info.items() if k != "power"},
        "iterations": int(losses.size),
        "figures": {
            "montage_phase": str(montage_phase),
            "montage_amplitude": str(montage_amp),
            "loss_plot": str(loss_png),
            "summary": str(summary) if summary else None,
        },
        "slice_thickness_A": thickness,
        "scan_step_A": sampling_a,
    }


def _object_sampling(cfg: PipelineConfig, output_dir: Path, stage: str) -> float:
    """Actual object pixel size in Angstrom, recorded by the worker."""
    mpath = output_dir / "stages" / stage / "metrics.json"
    if mpath.exists():
        s = json.loads(mpath.read_text(encoding="utf-8")).get("object_sampling_A")
        if s:
            return float(s)
    return float(cfg.experiment.scan_step_size)


def load_loss_histories(cfg: PipelineConfig, output_dir: Path) -> dict[str, np.ndarray]:
    histories: dict[str, np.ndarray] = {}
    for stage in cfg.stages:
        path = output_dir / "stages" / stage.name / "loss_history.npy"
        if path.exists():
            histories[stage.name] = np.load(path)
    return histories


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def execute_dag(
    cfg: PipelineConfig,
    config_path: Path,
    output_dir: Path,
    gpus: list[int],
    log_name: str = "gpu_assignment.log",
    metrics_name: str = "stage_metrics.json",
) -> tuple[bool, float, list[Job]]:
    pool = GpuPool(allowed=gpus, min_free_mb=8000)
    scheduler = GpuScheduler(pool, log_path=output_dir / log_name)
    jobs = build_jobs(cfg, config_path, output_dir)
    t0 = time.time()
    result = scheduler.run(jobs)
    wall = time.time() - t0
    write_stage_metrics(
        output_dir / metrics_name, jobs, collect_stage_metrics(output_dir, cfg)
    )
    return result.ok, wall, jobs


def run(cfg: PipelineConfig, config_path: Path, output_dir: Path, gpus: list[int]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(output_dir / "resolved_config.yaml")
    logger.info("%s", topology_report(cfg))
    (output_dir / "dag_topology.txt").write_text(
        topology_report(cfg), encoding="utf-8"
    )

    ok, wall, jobs = execute_dag(cfg, config_path, output_dir, gpus)
    metrics = collect_stage_metrics(output_dir, cfg)

    if not ok:
        logger.error("DAG aborted; see %s", output_dir / "gpu_assignment.log")
        write_final_report(output_dir, cfg, metrics, jobs, wall, None, None)
        return 1

    analysis = analyse_run(cfg, output_dir)
    write_final_report(output_dir, cfg, metrics, jobs, wall, analysis, None)
    print(json.dumps(analysis, indent=2))
    return 0


def collect_stage_metrics(output_dir: Path, cfg: PipelineConfig) -> dict:
    metrics: dict = {}
    for stage in cfg.stages:
        path = output_dir / "stages" / stage.name / "metrics.json"
        if path.exists():
            metrics[stage.name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


# --------------------------------------------------------------------------- #
# extension: continue the final stage and compare
# --------------------------------------------------------------------------- #
def run_extension(
    cfg: PipelineConfig,
    config_path: Path,
    output_dir: Path,
    gpus: list[int],
    total_iters: int,
) -> int:
    """Snapshot the current final stage, then continue it to ``total_iters``.

    ``total_iters`` is the *cumulative* iteration count for the stage, so a
    stage that already ran 50 iterations continues for ``total_iters - 50``
    more from its checkpoint.
    """
    final = cfg.stages[-1]
    base_iters = final.num_iters
    stage_dir = output_dir / "stages" / final.name

    existing = np.load(stage_dir / "loss_history.npy")
    base_iters = int(existing.size)
    if total_iters <= base_iters:
        logger.info(
            "stage %s already has %d iterations; nothing to extend",
            final.name,
            base_iters,
        )
        return 0
    increment = total_iters - base_iters

    # Keep the shorter result (and its figures) untouched for comparison.
    snap = output_dir / f"iter{base_iters}_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    shutil.copytree(stage_dir, snap / "stages" / final.name, dirs_exist_ok=True)
    for pattern in (
        "FINAL_phase_all_slices_4x4.png",
        "FINAL_amplitude_all_slices_4x4.png",
        "projected_phase_unwrapped.png",
        "projected_phase_complex_product.png",
        "phase_fft.png",
        "FINAL_projected_phase_unwrapped.npy",
        "loss_history.csv",
    ):
        src = output_dir / pattern
        if src.exists():
            shutil.copy2(src, snap / pattern)
    base_analysis = analyse_run(
        cfg, snap, prefix=f"ITER{base_iters}", write_shared_figures=False
    )

    # Continue the final stage from its OWN checkpoint so the iteration count is
    # cumulative (50 -> 100), not a fresh run from the upstream stage.
    ext_cfg = copy.deepcopy(cfg)
    ext_cfg.stage(final.name).num_iters = increment
    ext_cfg.stage(final.name).reset = False
    ext_path = output_dir / "resolved_config_ext.yaml"
    ext_cfg.to_yaml(ext_path)

    pool = GpuPool(allowed=gpus, min_free_mb=8000)
    scheduler = GpuScheduler(pool, log_path=output_dir / "gpu_assignment_ext.log")
    job = Job(
        name=f"{final.name}_iter{total_iters}",
        argv=[
            sys.executable, "-m", "ptycho.worker", "-c", str(ext_path),
            "--stage", final.name, "-o", str(output_dir),
            "--continue-from-self",
        ],
        deps=[],
        log_path=output_dir / "logs" / f"{final.name}_iter{total_iters}.log",
    )
    result = scheduler.run([job])
    if not result.ok:
        logger.error("extension run failed; snapshot preserved in %s", snap)
        return 1

    ext_analysis = analyse_run(cfg, output_dir, prefix="FINAL")
    comparison = {
        "stage": final.name,
        "base_iterations": base_iters,
        "extended_iterations": total_iters,
        "increment": increment,
        "base": base_analysis,
        "extended": ext_analysis,
        "sharpening": compare_phases(
            snap, output_dir, final.name, base_iters, total_iters
        ),
    }
    (output_dir / "iteration_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison["sharpening"], indent=2))
    return 0


def compare_phases(
    snap: Path, output_dir: Path, stage: str, base_iters: int, num_iters: int
) -> dict:
    """Quantify whether the extra iterations sharpened the atomic columns."""
    base = np.load(snap / "stages" / stage / "obj_cropped.npy")
    ext = np.load(output_dir / "stages" / stage / "obj_cropped.npy")

    out: dict = {}
    for label, obj in ((f"iter{base_iters}", base), (f"iter{num_iters}", ext)):
        phase, _, _ = decompose_object(obj)
        proj = projected_phase_unwrapped(phase)
        out[label] = phase_sharpness(proj)

    keys = ("phase_std", "variance", "mean_gradient_magnitude", "kurtosis",
            "high_freq_fraction", "peak_to_mean_fft")
    out["change_pct"] = {
        k: (
            100.0 * (out[f"iter{num_iters}"][k] - out[f"iter{base_iters}"][k])
            / out[f"iter{base_iters}"][k]
            if out[f"iter{base_iters}"][k]
            else None
        )
        for k in keys
    }
    return out


def phase_sharpness(proj: np.ndarray) -> dict:
    grad = np.gradient(proj)
    img = proj - proj.mean()
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(img * win))) ** 2
    ny, nx = power.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - ny // 2) ** 2 + (xx - nx // 2) ** 2)
    band = (r > 0.15 * min(ny, nx)) & (r < 0.45 * min(ny, nx))
    total = float(power.sum())
    return {
        "phase_std": float(proj.std()),
        "variance": float(proj.var()),
        "mean_gradient_magnitude": float(np.mean(np.hypot(grad[0], grad[1]))),
        "kurtosis": float(((proj - proj.mean()) ** 4).mean() / proj.var() ** 2)
        if proj.var() > 0
        else None,
        "high_freq_fraction": float(power[band].sum() / total) if total else 0.0,
        "peak_to_mean_fft": float(power[band].max() / power[band].mean())
        if band.any()
        else None,
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_final_report(
    output_dir: Path,
    cfg: PipelineConfig,
    metrics: dict,
    jobs: list[Job],
    wall: float,
    analysis: dict | None,
    comparison: dict | None,
) -> None:
    job_by_name = {j.name: j for j in jobs}
    # stage_metrics.json already merges the scheduler's real GPU id.
    merged_path = output_dir / "stage_metrics.json"
    merged = (
        json.loads(merged_path.read_text(encoding="utf-8"))
        if merged_path.exists()
        else {}
    )

    def gpu_of(name, m, job):
        for src in (merged.get(name, {}), m):
            if src.get("gpu") is not None:
                return src["gpu"]
        return job.gpu if job else None

    lines: list[str] = ["# PYREX atomic-phase reconstruction report", ""]
    lines.append(
        f"probe_energy = {cfg.experiment.probe_energy/1e3:.0f} keV -- "
        "**WARNING: assumed, not verified.** The H5 file has no attributes at all "
        "and no PYREX simulation config or script was found on disk."
    )
    lines.append(f"total wall time = {wall:.1f} s")
    lines.append(f"permitted GPUs = {sorted(set(j.gpu for j in jobs if j.gpu is not None))}")
    lines.append("")

    lines.append("## Stages")
    lines.append("")
    lines.append(
        "| stage | gpu | iters | initial loss | final loss | reduction % | runtime s | peak MiB | finite | status |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for stage in cfg.stages:
        m = metrics.get(stage.name, {})
        job = job_by_name.get(stage.name)
        init, fin = m.get("initial_loss"), m.get("final_loss")
        red = (
            100.0 * (init - fin) / init
            if init and fin is not None and init != 0
            else float("nan")
        )
        status = "ok" if (job and job.returncode == 0) else (
            "ok" if m else "MISSING"
        )
        lines.append(
            f"| {stage.name} | {gpu_of(stage.name, m, job)} "
            f"| {m.get('iterations_recorded', stage.num_iters)} "
            f"| {_fmt(init)} | {_fmt(fin)} | {red:.1f} "
            f"| {m.get('runtime_s', 0.0)} | {m.get('peak_memory_mb', 0.0)} "
            f"| {'yes' if m.get('finite') else 'NO'} | {status} |"
        )

    lines += ["", "## GPU assignment", ""]
    per_gpu: dict[int, list[str]] = {}
    for job in jobs:
        if job.gpu is not None:
            per_gpu.setdefault(job.gpu, []).append(job.name)
    for gpu in sorted(per_gpu):
        busy = sum(j.runtime for j in jobs if j.gpu == gpu and j.start and j.end)
        lines.append(f"- GPU {gpu}: {', '.join(per_gpu[gpu])} (busy {busy:.0f} s)")
    idle = [g for g in cfg_gpus(jobs) if g not in per_gpu]
    if idle:
        lines.append(f"- GPUs never used: {idle} -- the DAG is a chain with a single "
                     "2-way branch, so at most 2 stages can overlap.")

    if analysis:
        lines += ["", "## Object and phase", ""]
        lines.append(
            f"- object kind: `{analysis['object_kind']}` -- with `obj_type=potential` "
            "quantem stores a real array and forms the wave as `exp(1j*obj)`, so the "
            "stored value *is* the phase in radians (`np.angle` would return 0)."
        )
        lines.append(f"- object shape: {analysis['object_shape']}")
        lines.append(
            f"- per-slice phase: min={analysis['phase_min']:.4g} "
            f"max={analysis['phase_max']:.4g} std={analysis['phase_std']:.4g}"
        )
        lines.append(
            f"- projected phase (A): min={analysis['projected_phase_min']:.4g} "
            f"max={analysis['projected_phase_max']:.4g} "
            f"std={analysis['projected_phase_std']:.4g}"
        )
        lines.append(f"- projected phase (B): std={analysis['projected_phase_B_std']:.4g}")
        lines.append(
            f"- max |A - B| = {analysis['projected_phase_AB_max_diff']:.4g} rad"
        )
        fft = analysis["fft"]
        lines += ["", "## FFT periodicity", ""]
        lines.append(f"- peak-vs-background contrast: {fft['contrast']:.3f}")
        lines.append(f"- detected periodic peaks: {fft['num_peaks']}")
        lines.append(f"- implied d-spacing: {_fmt(fft['d_spacing_A'])} A")

        lines += ["", "## Figures", ""]
        for key, value in analysis["figures"].items():
            if value:
                lines.append(f"- {Path(value).resolve()}")
        for name in (
            "projected_phase_unwrapped.png",
            "projected_phase_complex_product.png",
            "projected_amplitude.png",
            "phase_fft.png",
            "loss_history.csv",
        ):
            lines.append(f"- {output_dir.resolve() / name}")

    lines += ["", "## Numerical sanity", ""]
    bad = [n for n, m in metrics.items() if not m.get("finite", True)]
    lines.append(
        f"- all object / probe / loss arrays finite: "
        f"{'YES' if not bad else 'NO -> ' + str(bad)}"
    )

    if comparison:
        lines += ["", "## 50 vs 100 iterations", ""]
        for k, v in comparison["sharpening"]["change_pct"].items():
            lines.append(f"- {k}: {_fmt(v)} %")

    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cfg_gpus(jobs: list[Job]) -> list[int]:
    return sorted({j.gpu for j in jobs if j.gpu is not None})


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.4g}"


def dag_complete(cfg: PipelineConfig, output_dir: Path) -> bool:
    """True when every stage already has a loss history and object array."""
    for stage in cfg.stages:
        sd = output_dir / "stages" / stage.name
        if not (sd / "loss_history.npy").exists():
            return False
        if not (sd / "obj_cropped.npy").exists():
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--gpus", default="5,6,7", help="comma-separated physical GPU ids")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-dag", action="store_true",
                        help="do not run the DAG; reuse existing stage results")
    parser.add_argument("--extend-final", type=int, default=None,
                        help="continue the final stage to this cumulative iteration count")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s | %(message)s"
    )

    cfg = PipelineConfig.from_yaml(args.config)
    cfg.validate()
    output_dir = (args.output_dir or Path(cfg.output.dir)).expanduser()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    forbidden = set(gpus) - PERMITTED_GPUS
    if forbidden:
        raise SystemExit(f"only GPUs {sorted(PERMITTED_GPUS)} are permitted; got {sorted(forbidden)}")

    if args.analyze_only:
        print(json.dumps(analyse_run(cfg, output_dir), indent=2))
        return 0

    # Never recompute a DAG whose results are already on disk.
    already_done = dag_complete(cfg, output_dir)
    if args.skip_dag or already_done:
        logger.info(
            "skipping DAG (%s); reusing stage results in %s",
            "--skip-dag" if args.skip_dag else "already complete",
            output_dir,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = collect_stage_metrics(output_dir, cfg)
        jobs: list[Job] = []
        wall = sum(m.get("runtime_s", 0.0) for m in metrics.values())
    else:
        rc = run(cfg, args.config, output_dir, gpus)
        if rc != 0:
            return rc
        metrics, jobs, wall = collect_stage_metrics(output_dir, cfg), [], 0.0

    if args.extend_final:
        return run_extension(cfg, args.config, output_dir, gpus, args.extend_final)

    analysis = analyse_run(cfg, output_dir)
    write_final_report(output_dir, cfg, metrics, jobs, wall, analysis, None)
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
