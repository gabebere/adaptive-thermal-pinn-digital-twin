# DeepXDE models and results

This folder owns all DeepXDE code, tests, and model-generated results in the
repository. The independent analytical literature reproduction and its
reference datasets live in `../Literature_results/`.

## Workflows

- `paper_data_model/` is the compact DeepXDE model that assimilates the
  literature paper's center-point reference values in sequential batches.
- `validated_adaptive_workflow/` is the larger full-field workflow containing
  analytical validation, offline PINN training, streaming adaptation,
  boundary-change experiments, and latency studies.

Each workflow has its own README, requirements, and results directory. Keep
new PINN outputs beside the workflow that generated them.
