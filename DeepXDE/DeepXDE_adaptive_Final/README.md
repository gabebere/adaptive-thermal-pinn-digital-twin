# DeepXDE adaptive final

This is the user-facing entry point for the maintained 2D adaptive PINN. It
does not contain multiple copies of the PINN. `run.py` calls the single tested
implementation in `core/`, while each TOML document
in `architectures/` supplies a different set of replaceable design choices.

## Run a preset

From this folder, using the repository virtual environment:

```powershell
..\..\..\.venv\Scripts\python.exe run.py balanced --profile full
```

Quick wiring check (reduced iterations and collocation points):

```powershell
..\..\..\.venv\Scripts\python.exe run.py balanced --profile smoke
```

Other maintained choices:

```powershell
..\..\..\.venv\Scripts\python.exe run.py low_latency --profile full
..\..\..\.venv\Scripts\python.exe run.py pytorch_comparison --profile full
```

You may also supply a file path:

```powershell
..\..\..\.venv\Scripts\python.exe run.py architectures\my_experiment.toml --profile full
```

Outputs are written below `outputs/` unless `--output-dir` is supplied. Every
run copies its exact TOML input to `architecture_used.toml` beside the metrics,
and the metrics JSON records the resolved architecture and training values.

## Change the architecture without copying the PINN

Copy one TOML document inside `architectures/`, give it a descriptive name,
and edit only that document. The accepted settings are:

- `[network]`: hidden-layer widths, activation, and weight initializer;
- `[training]`: offline/adaptive iterations, both learning rates, sensor-data
  and physics/data loss weights, plus PDE/boundary/initial collocation counts;
- `[streaming]`: sample spacing, update batch size, sensor coordinates, and
  recent observation-window length.

The loader is strict: unknown sections and misspelled setting names stop the
run with an error instead of being silently ignored. Omit
`observation_window_batches` to retain all observations.

The physical PDE, analytical Table 1 reference, unexpected boundary event,
and plotting code stay centralized in `core/`. This
keeps comparisons fair: an architecture document changes the model and its
online data policy, not the scientific problem underneath it.

## Included documents

- `balanced.toml`: the recommended compute/latency compromise and default;
- `low_latency.toml`: denser data and immediate updates, with much higher
  online cost;
- `pytorch_comparison.toml`: matches the standalone PyTorch model's 48x3 tanh
  network, Xavier/Glorot-uniform initialization, learning rates, training
  budget, collocation counts, and analogous sensor policy.

The PyTorch comparison does not change this workflow from 2D to 1D. It makes
the replaceable architecture/training choices comparable while preserving the
same validated 2D equation and reference data.

## Constant-flux PINN/PINO benchmark

The physical engine-wall benchmark and streamed PINO are retained alongside
the validated 2D workflow, but use a separate architecture document so their
training budgets cannot overwrite one another:

```powershell
..\..\..\.venv\Scripts\python.exe run_constant_flux.py constant_flux_balanced --profile full
```

`architectures/constant_flux_balanced.toml` defines the offline and adaptive
PINN architecture, the 8,001-step online adaptation budget, sensor locations,
loss weights, and collocation counts used by the constant-flux study. The
report-quality figures, checkpoints, metrics, and held-out cases remain under
`results/2026-07-22/`.

In short, `run.py balanced` runs the configurable validated 2D PINN, while
`run_constant_flux.py constant_flux_balanced` runs the physical constant-flux
PINN/PINO comparison. Both entry points load TOML architecture files.

## Multi-seed robustness study

The production-length five-seed benchmark is run with:

```powershell
..\..\..\.conda\python.exe core\run_multiseed_study.py --seeds 7 11 19 23 31 --skip-existing
```

It keeps the analytical corpus and held-out cases fixed while changing only
training randomness. Versioned aggregate results are under
`results/2026-07-22/multiseed_study/`; large per-seed checkpoints remain in the
ignored `outputs/multiseed_study/` directory.
