# DeepXDE adaptive final

This is the user-facing entry point for the maintained 2D adaptive PINN. It
does not contain multiple copies of the PINN. `run.py` calls the single tested
implementation in `core/`, while each TOML document
in `architectures/` supplies a different set of replaceable design choices.

## Run a preset

From this folder, using the repository virtual environment:

```powershell
..\..\.venv\Scripts\python.exe run.py balanced --profile full
```

Quick wiring check (reduced iterations and collocation points):

```powershell
..\..\.venv\Scripts\python.exe run.py balanced --profile smoke
```

Other maintained choices:

```powershell
..\..\.venv\Scripts\python.exe run.py low_latency --profile full
..\..\.venv\Scripts\python.exe run.py pytorch_comparison --profile full
```

You may also supply a file path:

```powershell
..\..\.venv\Scripts\python.exe run.py architectures\my_experiment.toml --profile full
```

Outputs are written below `outputs/` unless `--output-dir` is supplied. Every
run copies its exact TOML input to `architecture_used.toml` beside the metrics,
and the metrics JSON records the resolved architecture and training values.

## Change the architecture without copying the PINN

Copy one TOML document inside `architectures/`, give it a descriptive name,
and edit only that document. The accepted settings are:

- `[network]`: hidden-layer widths, activation, and weight initializer;
- `[training]`: offline/adaptive iterations, both learning rates, sensor-data
  and physics/data loss weights, PDE/boundary/initial collocation counts,
  point distribution, and optional PDE-point resampling policy;
- `[streaming]`: sample spacing, update batch size, sensor coordinates, and
  recent observation-window length. A preset may alternatively request an
  exact number of adaptive windows, exclude the known initial-time sample,
  and add deterministic sensor noise.

The loader is strict: unknown sections and misspelled setting names stop the
run with an error instead of being silently ignored. Omit
`observation_window_batches` to retain all observations.

The physical PDE, analytical Table 1 reference, unexpected boundary event,
and plotting code stay centralized in `core/`. This
keeps comparisons fair: an architecture document changes the model and its
online data policy, not the scientific problem underneath it.

The maintained workflow uses continuous left-edge heating
`theta(0,Y,tau)=Y(1-Y)` with zero temperature on the other three edges, so the
reference approaches a nonzero steady field. Figure 06 applies an unannounced
50% increase in this heating amplitude halfway through the run. The literature
validation stage remains a separate check of the original decaying paper cases.

## Included documents

- `balanced.toml`: the recommended compute/latency compromise and default;
- `low_latency.toml`: denser data and immediate updates, with much higher
  online cost;
- `pytorch_comparison.toml`: matches the standalone PyTorch model's 48x3 tanh
  network, Xavier/Glorot-uniform initialization, learning rates, training
  budget, collocation counts, and analogous sensor policy.

The PyTorch comparison additionally uses pseudorandom collocation points,
resamples interior/initial physics points every Adam step, uses 512 total 2-D
boundary samples to match the PyTorch model's 512 boundary evaluations, splits
40 post-initial sensor times into five equal windows, and applies 1% normalized
sensor noise. Boundary resampling remains disabled because DeepXDE requires
stable per-edge counts for the four separately weighted conditions.

The PyTorch comparison does not change this workflow from 2D to 1D. It makes
the replaceable architecture/training choices comparable while preserving the
same validated 2D equation and reference data.
