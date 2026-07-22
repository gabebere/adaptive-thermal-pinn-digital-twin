# Five-seed training-stability study

This study retrains the offline PINN, sensor-adaptive PINN, and streamed PINO
with seeds `7`, `11`, `19`, `23`, and `31`. Every run uses the same
`constant_flux_balanced.toml` architecture, the same analytical CSV corpus, and
the same locked 4.0 to 5.2 MW m⁻² heat-flux step at `tau=50`. Only stochastic
training state changes between runs.

The fixed corpus manifest SHA-256 is
`ad103141b308e072444087f66ed0064a4c0014ad6cdca78db659d9892de33383`.
PINO validation selects the lowest-validation-RMSE checkpoint for every seed;
the test cases never select checkpoints.

## Locked-case overall RMSE

| Model | Mean RMSE (K) | Sample standard deviation (K) | Range (K) | CV |
|---|---:|---:|---:|---:|
| Offline PINN | 19.502 | 1.441 | 17.554–21.271 | 7.39% |
| Sensor-adaptive PINN | 2.208 | 0.375 | 1.921–2.825 | 16.99% |
| Streamed PINO | **0.233** | **0.006** | **0.222–0.238** | **2.60%** |

Across seeds, the streamed PINO has 9.49 times lower mean RMSE than the
sensor-adaptive PINN and 83.81 times lower mean RMSE than the offline PINN. Its
mean RMSE over the 200 held-out interpolation cases is `0.249 +/- 0.006 K`
across training seeds. This supports both superior accuracy and substantially
lower seed sensitivity for the tested in-family boundary-condition distribution.

## Files

- `01_seedwise_locked_case_rmse.png`: seed-by-seed comparison of all models;
- `02_locked_case_rmse_distribution.png`: model RMSE distributions;
- `03_pino_generalization_by_seed.png`: PINO locked, validation, and 200-case
  held-out interpolation errors;
- `seedwise_metrics.csv`: all primary model metrics by seed;
- `pino_seedwise_generalization.csv`: PINO validation/test metrics and selected
  epochs;
- `summary.json`: means, sample standard deviations, ranges, coefficients of
  variation, latency, corpus hash, and improvement factors;
- `seed_metrics/`: complete metrics for each independent run.

The trained seed checkpoints and pointwise prediction arrays remain in the
ignored `outputs/multiseed_study/` directory; they are intentionally excluded
from Git because of their size. The study can be resumed without overwriting
completed seeds:

```powershell
..\..\..\.conda\python.exe core\run_multiseed_study.py --seeds 7 11 19 23 31 --skip-existing
```
