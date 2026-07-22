# Constant-flux adaptive PINN / streamed PINO benchmark

This is the maintained entry point for the physical engine-wall comparison.
It runs the exact folder linked on GitHub; the DeepXDE code has not been copied
into a separate DeepONet working folder.

## Physical problem

The learned models solve a one-dimensional wall of thickness `L=5 mm`:

```text
T_t = alpha T_xx
-k T_x(0,t) = q_hot(t)                  (constant engine-side heat flux)
-k T_x(L,t) = h [T(L,t) - T_coolant]    (convective coolant side)
T(x,0) = T_coolant = 300 K
```

The locked test applies `4.0 MW/m²` up to `t=0.5 s` and `5.2 MW/m²`
afterward. The flux does not decay. Temperature is continuous at the switch.
The reference is an exact Robin-eigenfunction series; the switched response is
constructed by causal step-response superposition, rather than by splicing two
independent temperature histories.

The workflow also reruns the original dimensionless 2-D paper-series check
against its published tables. That check is kept separate because the paper's
2-D Dirichlet problem and this physical 1-D Neumann/Robin wall are different
PDE boundary-value problems; their errors must never be mixed.

## Models compared

- **Offline baseline PINN:** a DeepXDE FNN trained once using the pre-switch
  `4.0 MW/m²` boundary condition. It receives no streamed observations.
- **Adaptive balanced PINN:** the same DeepXDE FNN warm-started from the
  offline weights, then updated in two-sample batches from three sensor streams
  and the known flux history (retaining the two latest batches). The balanced preset uses 21 updates ×
  381 iterations = 8,001 adaptive optimization steps.
- **Streamed PINO:** a GRU observer consumes the current three temperatures,
  hot-side flux, flux increment, time, and sample interval. Its state conditions
  a Fourier-feature spatial decoder. Training uses 500 epochs and combines exact
  CSV field supervision (weighted around boundary events) with heat-equation and
  boundary residual losses; deployment performs one forward state update without
  online gradient descent. Validation selected epoch 350 from the full run.

## Reproduce the full run

From this directory:

```powershell
& '..\..\..\.conda\python.exe' run.py balanced --profile full --output-dir outputs
```

Use `--profile smoke` only to check wiring. It deliberately has too little
training to be used as a performance result.

The improved deterministic analytical corpus is stored in `data/mixed_flux_v2/`.
It contains 600 training, 100 validation, 200 held-out interpolation, and one
locked switch scenario as CSV files. Event times and jump magnitudes are
stratified, increases and decreases are balanced, and 30% of the cases relax
exponentially after the event while 70% remain non-decaying steps. The manifest,
checksums, and exact generation configuration let another model use identical
data. The locked comparison remains the physical non-decaying 4.0 to 5.2 MW/m²
case, so extra generality cannot hide a regression on the target problem.

The report-quality results are in
`outputs/mixed_flux_balanced/full/`, including:

- `graphs/00a_paper_analytical_validation.png`;
- `graphs/00b_physical_analytical_solution.png`;
- `graphs/00c_training_corpus_coverage.png`;
- `graphs/01_balanced_adaptive_vs_baseline_error.png`;
- `graphs/02_all_models_temperature_values.png`;
- `graphs/03_adaptive_pinn_vs_streamed_pino_error.png`;
- `graphs/04_pino_before_after_improvement.png`;
- `metrics.json`, `rmse_by_time.csv`, model checkpoints, and the complete
  point-by-point prediction arrays.

## Reuse without retraining

The small, curated model files in `checkpoints/` are intentionally tracked in
Git even though ordinary `.pt` files and generated `outputs/` are ignored. The
checkpoint manifest records their hashes and roles. New evaluations load the
offline PINN and streamed PINO directly; only the adaptive PINN's defining
online sensor-update phase is rerun for each new boundary history.

## Published result snapshot

The report-quality graphs, metrics, point-by-point predictions, and packaged
held-out test data from this study are versioned under `results/2026-07-22/`.
Its README documents the cases, model provenance, boundary conditions, archive
checksum, and headline RMSE values.
