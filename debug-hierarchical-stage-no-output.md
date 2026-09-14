[OPEN] hierarchical-stage-no-output

## Symptom
- User reports: running `python single_slice_geometry_hierarchical_300kev.py` on server shows “no movement/no output”.

## Expected
- If `--stage` is missing, argparse should print usage and exit quickly.
- If `--stage A` is provided, should start Stage A jobs and write logs/artifacts under:
  `runs/single_slice_geometry_hierarchical_300kev/`.

## Hypotheses (falsifiable)
1. The command was run without `--stage`, so argparse exits immediately; user misinterprets as “no movement”.
2. The script prints usage but output is not visible (terminal scrollback / redirected stdout/stderr).
3. The script is hanging during imports (e.g., numpy / matplotlib backend / quantem import indirectly).
4. Stage A was started but appears silent because it logs primarily to files; progress exists in `gpu_assignment.log` and per-run `worker.log`.
5. Scheduler is blocked on `nvidia-smi` query (command hangs), so no jobs start.

## Evidence to collect
- `python single_slice_geometry_hierarchical_300kev.py -h` output
- `python single_slice_geometry_hierarchical_300kev.py` output/exit code
- `python single_slice_geometry_hierarchical_300kev.py --stage A` creates:
  - `runs/single_slice_geometry_hierarchical_300kev/gpu_assignment.log`
  - `runs/single_slice_geometry_hierarchical_300kev/stageA_job_summary.json`

## Evidence observed (this workspace)
- `gpu_assignment.log` contains only the initial line:
  - "scheduler start: allowed GPUs=[5, 6, 7] ..."
  - no subsequent START/DONE/FAILED lines
  -> consistent with the scheduler blocking before spawning the first job (likely during GPU query).

## Next actions
- Reproduce locally with `-h`, no-arg, and `--stage A` runs.
- If hang: isolate whether hang is in argparse, import, or `nvidia-smi`.
