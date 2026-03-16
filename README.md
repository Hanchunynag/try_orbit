# LEO SGP4 vs Orekit Numerical Truth Residual Learning

## What Changed

This project no longer uses the previous hand-written `two-body + J2 + optional drag` propagator as the main synthetic truth generator.

The current implementation uses:

- **Orekit TLEPropagator** for the nominal SGP4 orbit
- **Orekit NumericalPropagator** for the synthetic truth orbit

This is the correct direction if the goal is to generate a more realistic open-source high-fidelity reference trajectory instead of a simplified custom Cowell approximation.

## Physical Meaning

This experiment still follows the same important interpretation:

- The TLE is **not** a real precision truth orbit.
- The TLE is first used to initialize the **nominal SGP4 orbit**.
- The **SGP4 Cartesian state at the TLE epoch** is then used as the initial condition of the **Orekit numerical propagator**.
- Therefore, the Orekit numerical orbit is still a **synthetic reference truth**, not a real observed orbit.

The model learns the time evolution of the residual between:

- nominal SGP4 from TLE
- Orekit numerical synthetic truth

and then extrapolates that residual to correct SGP4 over a long arc.

## Reference Frame

All comparisons are performed in the same inertial frame:

- **GCRF**

The nominal SGP4 trajectory is generated with Orekit `TLEPropagator` and queried directly in GCRF.
The numerical truth trajectory is propagated by Orekit `NumericalPropagator` and sampled in GCRF.

So the previous TEME-to-ECI mismatch risk is removed from the main pipeline.

## Force Models in the Current Orekit Truth Propagator

The synthetic truth is now generated with Orekit numerical propagation using configurable force models.
Default configuration includes:

- Earth spherical harmonics gravity via `HolmesFeatherstoneAttractionModel`
- Gravity field degree/order configurable, default `21 x 21`
- Sun and Moon third-body perturbations
- NRLMSISE-00 atmospheric drag
- Solar radiation pressure
- Optional relativity correction

This is substantially closer to a realistic open-source HPOP-style workflow than the previous custom model.

## Important Limitation

Even with Orekit numerical propagation, this is still **not** equivalent to a commercial operational HPOP product unless you also provide suitable:

- force model selection
- gravity field degree/order
- atmospheric and solar activity data
- spacecraft physical parameters
- model tuning consistent with your mission object

So the project now uses a much stronger open-source truth generator, but it remains a **synthetic truth experiment**, not a certified truth orbit determination system.

## Orekit Data Requirement

Orekit requires an external `orekit-data` dataset for gravity field, Earth orientation, solar activity, and related models.

You must provide either:

- `--orekit_data_path <path-to-orekit-data-dir-or-zip>`
- or environment variable `OREKIT_DATA_PATH`

The code also searches common local names such as:

- `./orekit-data`
- `./orekit-data.zip`
- `~/orekit-data`
- `~/orekit-data.zip`

If none is found, the program raises a clear error.

## Installation

```bash
pip install -r requirements.txt
```

## Main Dependencies

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `torch`
- `sgp4`
- `jpype1`
- `orekit-jpype`

## Quick Start

```bash
python main.py \
  --tle_file data/sample.tle \
  --sat_name "ISS (ZARYA)" \
  --orekit_data_path D:\path\to\orekit-data \
  --train_duration 400 \
  --total_duration 36000 \
  --dt_train 0.01 \
  --dt_full 0.1 \
  --model_name tcn \
  --input_len 500 \
  --pred_len 100 \
  --noise_sigma_r 1.0 \
  --noise_sigma_t 1.0 \
  --noise_sigma_n 1.0 \
  --orekit_enable_third_body \
  --orekit_enable_drag \
  --orekit_enable_srp \
  --seed 42
```

## Paper-Style NARX Sweep

Use `sweep_narx.py` to run a random search over the paper-style NARX hyperparameters:

- activation function
- number of hidden layers
- nodes per layer
- prediction length in seconds
- optimizer

The sweep script forwards the remaining arguments to `main.py`, creates one output directory per trial, and writes:

- `manifest.json`
- `results.json`
- `results.csv`
- `best_trial.json`

Example:

```bash
python sweep_narx.py \
  --sweep_num_trials 24 \
  --sweep_output_dir outputs/sweeps/narx_table1 \
  --tle_file data/sample.tle \
  --sat_name "ISS (ZARYA)" \
  --orekit_data_path D:\path\to\orekit-data \
  --train_duration 400 \
  --total_duration 3600 \
  --dt_train 0.1 \
  --dt_full 0.1 \
  --observation_mode residual \
  --noise_sigma_r 0.0 \
  --noise_sigma_t 0.0 \
  --noise_sigma_n 0.0 \
  --seed 42
```

## Your Scenario Example

```bash
python main.py \
  --tle_file "D:\研究生\低轨卫星定位尝试\小论文VARPRO+REML\tle.tle" \
  --sat_name "YOUR SAT NAME" \
  --scenario_start_utc 2026-01-15T03:30:00Z \
  --receiver_lat 45.772625 \
  --receiver_lon 126.682625 \
  --receiver_alt 154 \
  --orekit_data_path "D:\orekit-data" \
  --train_duration 400 \
  --total_duration 36000 \
  --dt_train 0.01 \
  --dt_full 0.1 \
  --model_name tcn \
  --input_len 500 \
  --pred_len 100 \
  --orekit_enable_third_body \
  --orekit_enable_drag \
  --orekit_enable_srp \
  --seed 42
```

## CLI Parameters Added for Orekit

- `--orekit_data_path`
- `--orekit_mass_kg`
- `--orekit_drag_cd`
- `--orekit_drag_area_m2`
- `--orekit_srp_cr`
- `--orekit_srp_area_m2`
- `--orekit_gravity_degree`
- `--orekit_gravity_order`
- `--orekit_enable_third_body`
- `--orekit_enable_drag`
- `--orekit_enable_srp`
- `--orekit_enable_relativity`
- `--orekit_solar_activity_level`
- `--orekit_min_step_sec`
- `--orekit_max_step_sec`
- `--orekit_abs_tolerance`
- `--orekit_rel_tolerance`

## RTN Definition

RTN is still defined strictly from the **nominal SGP4 orbit**, not from the numerical truth orbit:

- `R`: nominal SGP4 radial direction
- `N`: nominal SGP4 angular momentum direction
- `T`: `N x R`

Residuals are then projected and reconstructed in that SGP4-defined RTN frame.

## Training / Forecast Isolation

Time separation is still strict:

- training: `0 <= t < 400 s`
- forecast test: `400 <= t <= total_duration`

No future numerical-truth information is used during rollout.

## Outputs

The run saves:

- `outputs/data/sgp4_states.csv`
- `outputs/data/hpop_states.csv`
- `outputs/data/corrected_states.csv`
- `outputs/data/residual_clean.csv`
- `outputs/data/residual_noisy_train.csv`
- `outputs/data/scaler.json`
- `outputs/models/best_model.pt`
- `outputs/metrics/metrics.json`
- `outputs/metrics/metrics.csv`
- `outputs/figures/*.png`
- `outputs/used_config.json`
- `outputs/run.log`

`hpop_states.csv` is kept as the output filename for compatibility with the previous workflow, but its contents are now generated by **Orekit NumericalPropagator**, not by the old custom propagator.

## Memory / Runtime Notes

The Orekit version is heavier than the previous simplified propagator.

Recommendations:

- keep `dt_train = 0.01`
- use `dt_full = 0.1` for first runs
- enable only the force models you need at first
- start with gravity `21 x 21`
- increase gravity order only after you confirm runtime is acceptable

Very fine `dt_full` together with strong force-model settings will significantly increase runtime.

## Current Scope

Still single-satellite only.
Ground-station LLA is currently metadata only and is not yet used for line-of-sight filtering or actual measurement geometry.
