# Dust + Muon experiments

Run from the repository root on the benchmark's 8xH100 machine, after `uv sync`
and data preparation (`uv run utils/prepare_data.py`).

```bash
# Fixed 1,831 updates (one epoch), Adam and Muon sequentially.
# Also records time to target if reached during either run.
bash train/dust_muon_runs.sh fixed

# Dedicated speedruns: stop each optimizer at its first measured loss <= 3.5.
bash train/dust_muon_runs.sh speedrun
```

Both use seed 0, batch 8, population scale 1.5, and 30% LR warmdown over the
planned step budget. A shorter `STEPS` also shortens that schedule. Every run
starts fresh; output directories must be empty. Defaults use distinct directories
under `logs/dust-muon/{fixed,speedrun}/{adam,muon}`.

Muon uses the block attention/MLP projections, momentum 0.95, Nesterov,
five Newton–Schulz iterations, original aspect-ratio LR scaling, LR 0.01,
and zero weight decay. Adam retains the existing rates for embeddings, head,
gates and mixing scalars. Gradients are averaged across ranks before momentum.
Both optimizers' steps use `torch.compile` by default. Tensor learning rates
allow warmdown without specializing a new graph per rate. Untimed warmup covers
both fresh and initialized optimizer state, then rebuilds training from seed 0.
Use `--eager-optimizer` on `dust.py` for an eager comparison. Check compiled/eager
updates and schedule recompilation with
`uv run train/check_dust_optimizer.py --device cuda`.

For a smoke run or an LR sweep:

```bash
STEPS=2 EVAL_EVERY=1 OUTPUT_ROOT=logs/dust-muon/smoke bash train/dust_muon_runs.sh fixed
MUON_LR=0.005 OUTPUT_ROOT=logs/dust-muon/lr-005 bash train/dust_muon_runs.sh fixed
MUON_LR=0.02 OUTPUT_ROOT=logs/dust-muon/lr-020 bash train/dust_muon_runs.sh fixed
```

Other overrides: `GPUS`, `SCALE`, `DATA_DIR`, `TARGET_LOSS`. Supported GPU counts
must satisfy Dust's population divisibility checks. Smaller populations change
the estimator's noise and should be compared with a matched Adam run.

## Reading results

- **Fixed budget:** compare `result.json`'s final `val_loss` at the same `steps`.
  No intermediate checkpoint is selected.
- **Speed:** compare `target_hit.train_seconds` and `target_hit.step`.
  `target_hit: null` means the target was not reached at a validation check.
- `metrics.jsonl` records loss/time at every update and validation every 100
  updates, plus the last update. Target time is the first *observed* crossing;
  use the same evaluation cadence for comparisons.
- Timing excludes warmup and validation, following the benchmark. It includes
  training and optimizer work; it is not total job wall time. The fixed-budget
  run provides both metrics, so a separate speedrun is optional.

The existing Dust/Adam record is 3.767, above the speedrun target of 3.5.
These runs test whether Muon can close that gap; reaching 3.5 is not assumed.

## Obsidian

`objob.toml` requests 8 H100s on `cgpus` for four hours. The initial comparison
uses 400 updates per optimizer: the four-GPU smoke test took roughly 45–60
seconds per update, so even with ideal scaling, 1,000 updates on eight GPUs
would exceed the partition limit. Data lives in `/home/jonathan.mei/data/noise`.
Submit each optimizer separately so each has its own four-hour allocation.
Both runs measure final loss and time to the first observed target crossing.

```bash
objob submit --partition cpus --gres '' --cpus 8 --mem 32G --job-name noise-data -- \
  uv run utils/prepare_data.py --output /home/jonathan.mei/data/noise
# Once data preparation and the two-step GPU smoke test succeed:
objob submit --job-name dust-muon-compiled-400
objob submit --job-name dust-adam-compiled-400 -- \
  uv run torchrun --standalone --nproc_per_node=8 train/val/dust.py \
  --optimizer adam --steps 400 --eval-every 100 --target-loss 3.5 \
  --data-dir /home/jonathan.mei/data/noise --output logs/obsidian-adam-compiled-400
```
