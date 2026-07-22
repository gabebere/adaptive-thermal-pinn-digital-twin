# Expanded analytical reference results

This folder contains analytical and numerical-reference artifacts that extend
the eight published study times used by the literature reproduction.

## `long_horizon_tau_100/`

- `02_reference_field.png` visualizes the validated analytical reference over
  the full configured horizon, ending at dimensionless time `tau = 100`.
- `02_validated_reference_dataset.npz` contains the corresponding full-field
  reference dataset.
- `02_validated_reference_dataset.csv` contains the same 11,849 full-field
  values in readable `x`, `y`, `tau`, and `temperature` columns.

## `combined_boundary_change/`

- `06_boundary_change_reference.npz` contains the numerical reference for the
  combined boundary-condition experiment, with boundary set 1 followed by
  boundary set 2.
- `06_boundary_change_reference.csv` contains the same full-field reference in
  readable columns: `x`, `y`, `tau`, `temperature`, and `is_post_switch`.
- `06_boundary_change_response.png` visualizes that reference alongside the
  adaptive-model response.

The source implementations remain in
`../../DeepXDE/validated_adaptive_workflow/` because they are executable
dependencies of that workflow. These copies are the literature/reference
artifacts; PINN-only predictions and error studies remain under `DeepXDE/`.
