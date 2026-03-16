"""Configuration management for the orbit residual learning experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """Runtime configuration for the end-to-end experiment."""

    tle_file: str
    sat_name: Optional[str]
    output_dir: str = "outputs"
    precomputed_bundle_path: Optional[str] = None
    scenario_start_utc: Optional[str] = None
    orekit_data_path: Optional[str] = None
    receiver_lat_deg: float = 45.772625
    receiver_lon_deg: float = 126.682625
    receiver_alt_m: float = 154.0
    train_duration_sec: float = 400.0
    total_duration_sec: float = 36000.0
    dt_train_sec: float = 0.01
    dt_full_sec: float = 0.1
    sgp4_chunk_size: int = 5000
    hpop_chunk_size: int = 2000
    observation_mode: str = "residual"
    predict_velocity: bool = False
    noise_sigma_r_m: float = 1.0
    noise_sigma_t_m: float = 1.0
    noise_sigma_n_m: float = 1.0
    noise_sigma_vr_mps: float = 0.01
    noise_sigma_vt_mps: float = 0.01
    noise_sigma_vn_mps: float = 0.01
    model_name: str = "narx"
    input_len: int = 500
    pred_len: int = 100
    window_stride: int = 1
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 8
    lambda_pred: float = 1.0
    lambda_diff: float = 0.2
    lambda_smooth: float = 0.05
    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    tcn_kernel_size: int = 3
    narx_input_lags: int = 1
    narx_feedback_lags: int = 1
    narx_hidden_dim: int = 10
    narx_num_hidden_layers: int = 1
    narx_activation: str = "tanh"
    narx_dropout: float = 0.0
    narx_snake_alpha: float = 1.0
    narx_prediction_length_sec: float = 0.1
    narx_use_velocity_input: bool = False
    optimizer_name: str = "adam"
    optimizer_momentum: float = 0.9
    lr_scheduler_name: str = "plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    lr_scheduler_min_lr: float = 1e-6
    orekit_mass_kg: float = 10.0
    orekit_drag_cd: float = 2.2
    orekit_drag_area_m2: float = 0.1
    orekit_srp_cr: float = 1.2
    orekit_srp_area_m2: float = 0.1
    orekit_gravity_degree: int = 21
    orekit_gravity_order: int = 21
    orekit_enable_third_body: bool = True
    orekit_enable_drag: bool = True
    orekit_enable_srp: bool = True
    orekit_enable_relativity: bool = False
    orekit_solar_activity_level: str = "AVERAGE"
    orekit_min_step_sec: float = 1e-3
    orekit_max_step_sec: float = 120.0
    orekit_abs_tolerance: float = 1e-3
    orekit_rel_tolerance: float = 1e-12
    memmap_threshold_mb: float = 256.0
    csv_float_format: str = "%.9f"

    def to_dict(self) -> dict:
        """Convert config to a JSON-serializable dictionary."""
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="LEO SGP4 vs Orekit numerical truth residual learning experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tle_file", required=True, help="Path to the TLE file.")
    parser.add_argument("--sat_name", default=None, help="Satellite name to select from TLE.")
    parser.add_argument("--output_dir", default="outputs", help="Output directory.")
    parser.add_argument(
        "--precomputed_bundle_path",
        default=None,
        help="Optional NPZ bundle with precomputed SGP4/HPOP/RTN residual data. Useful for sweeps that reuse the same scenario.",
    )
    parser.add_argument(
        "--scenario_start_utc",
        default=None,
        help="Observation start time in UTC, e.g. 2026-01-15T03:30:00. If omitted, TLE epoch is used.",
    )
    parser.add_argument(
        "--orekit_data_path",
        default=None,
        help="Path to orekit-data directory or zip. If omitted, common local paths and OREKIT_DATA_PATH are searched.",
    )
    parser.add_argument("--receiver_lat", type=float, default=45.772625, help="Receiver latitude in degrees.")
    parser.add_argument("--receiver_lon", type=float, default=126.682625, help="Receiver longitude in degrees.")
    parser.add_argument("--receiver_alt", type=float, default=154.0, help="Receiver altitude in meters.")
    parser.add_argument("--train_duration", type=float, default=400.0, help="Training duration in seconds.")
    parser.add_argument("--total_duration", type=float, default=36000.0, help="Total scenario duration in seconds.")
    parser.add_argument("--dt_train", type=float, default=0.01, help="Dense sampling interval for training segment in seconds.")
    parser.add_argument("--dt_full", type=float, default=0.1, help="Sampling interval for forecast segment in seconds.")
    parser.add_argument("--observation_mode", choices=["residual", "state"], default="residual")
    parser.add_argument(
        "--predict_velocity",
        action="store_true",
        help="Learn both position and velocity RTN residuals. Leave disabled to learn only RTN position residuals [R, T, N], which is recommended for short observation arcs.",
    )
    parser.add_argument("--noise_sigma_r", type=float, default=1.0, help="Noise sigma in R direction, meters.")
    parser.add_argument("--noise_sigma_t", type=float, default=1.0, help="Noise sigma in T direction, meters.")
    parser.add_argument("--noise_sigma_n", type=float, default=1.0, help="Noise sigma in N direction, meters.")
    parser.add_argument("--noise_sigma_vr", type=float, default=0.01, help="Velocity noise sigma in R, m/s.")
    parser.add_argument("--noise_sigma_vt", type=float, default=0.01, help="Velocity noise sigma in T, m/s.")
    parser.add_argument("--noise_sigma_vn", type=float, default=0.01, help="Velocity noise sigma in N, m/s.")
    parser.add_argument("--model_name", choices=["narx", "tcn", "lstm"], default="narx")
    parser.add_argument("--input_len", type=int, default=500, help="History window length.")
    parser.add_argument("--pred_len", type=int, default=100, help="Prediction horizon per rollout block.")
    parser.add_argument("--window_stride", type=int, default=1, help="Sliding-window stride.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--lambda_pred", type=float, default=1.0)
    parser.add_argument("--lambda_diff", type=float, default=0.2)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument(
        "--narx_input_lags",
        type=int,
        default=1,
        help="Number of delayed exogenous SGP4 input samples used by the NARX predictor.",
    )
    parser.add_argument(
        "--narx_feedback_lags",
        type=int,
        default=1,
        help="Number of delayed residual observations/predictions fed back into the NARX predictor.",
    )
    parser.add_argument(
        "--narx_hidden_dim",
        type=int,
        default=10,
        help="Hidden layer width of the paper-style NARX predictor.",
    )
    parser.add_argument(
        "--narx_num_hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in the paper-style NARX predictor.",
    )
    parser.add_argument(
        "--narx_activation",
        choices=["linear", "relu", "tanh", "sigmoid", "snake"],
        default="tanh",
        help="Hidden activation of the paper-style NARX predictor.",
    )
    parser.add_argument(
        "--narx_dropout",
        type=float,
        default=0.0,
        help="Dropout inside the paper-style NARX predictor.",
    )
    parser.add_argument(
        "--narx_snake_alpha",
        type=float,
        default=1.0,
        help="Snake activation alpha parameter used when narx_activation=snake.",
    )
    parser.add_argument(
        "--narx_prediction_length",
        type=float,
        default=0.1,
        help="Prediction horizon in seconds for the NARX target, matching the paper's prediction-length hyperparameter.",
    )
    parser.add_argument(
        "--narx_use_velocity_input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Augment NARX exogenous inputs with nominal SGP4 velocity states. Default keeps the paper's position-focused setup.",
    )
    parser.add_argument(
        "--optimizer_name",
        choices=["adam", "adagrad", "sgd", "yogi", "adamw"],
        default="adam",
        help="Optimizer used for model training. The first four options align with the paper's Table 1 search space.",
    )
    parser.add_argument("--optimizer_momentum", type=float, default=0.9, help="Momentum used by SGD.")
    parser.add_argument(
        "--lr_scheduler_name",
        choices=["none", "plateau"],
        default="plateau",
        help="Learning-rate scheduler. Plateau reduction is closer to the paper's incremental LR decay.",
    )
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5, help="LR decay factor for the plateau scheduler.")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5, help="Patience for the plateau scheduler.")
    parser.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6, help="Minimum LR for the plateau scheduler.")
    parser.add_argument("--orekit_mass_kg", type=float, default=10.0)
    parser.add_argument("--orekit_drag_cd", type=float, default=2.2)
    parser.add_argument("--orekit_drag_area_m2", type=float, default=0.1)
    parser.add_argument("--orekit_srp_cr", type=float, default=1.2)
    parser.add_argument("--orekit_srp_area_m2", type=float, default=0.1)
    parser.add_argument("--orekit_gravity_degree", type=int, default=21)
    parser.add_argument("--orekit_gravity_order", type=int, default=21)
    parser.add_argument("--orekit_enable_third_body", action=argparse.BooleanOptionalAction, default=True, help="Enable Sun/Moon third-body attraction.")
    parser.add_argument("--orekit_enable_drag", action=argparse.BooleanOptionalAction, default=True, help="Enable NRLMSISE00 atmospheric drag.")
    parser.add_argument("--orekit_enable_srp", action=argparse.BooleanOptionalAction, default=True, help="Enable solar radiation pressure.")
    parser.add_argument("--orekit_enable_relativity", action=argparse.BooleanOptionalAction, default=False, help="Enable relativistic correction.")
    parser.add_argument("--orekit_solar_activity_level", choices=["AVERAGE", "WEAK", "STRONG"], default="AVERAGE")
    parser.add_argument("--orekit_min_step_sec", type=float, default=1e-3)
    parser.add_argument("--orekit_max_step_sec", type=float, default=120.0)
    parser.add_argument("--orekit_abs_tolerance", type=float, default=1e-3)
    parser.add_argument("--orekit_rel_tolerance", type=float, default=1e-12)
    parser.add_argument("--sgp4_chunk_size", type=int, default=5000)
    parser.add_argument("--hpop_chunk_size", type=int, default=2000)
    parser.add_argument("--memmap_threshold_mb", type=float, default=256.0)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create the runtime config from parsed CLI args."""
    return ExperimentConfig(
        tle_file=args.tle_file,
        sat_name=args.sat_name,
        output_dir=args.output_dir,
        precomputed_bundle_path=args.precomputed_bundle_path,
        scenario_start_utc=args.scenario_start_utc,
        orekit_data_path=args.orekit_data_path,
        receiver_lat_deg=args.receiver_lat,
        receiver_lon_deg=args.receiver_lon,
        receiver_alt_m=args.receiver_alt,
        train_duration_sec=args.train_duration,
        total_duration_sec=args.total_duration,
        dt_train_sec=args.dt_train,
        dt_full_sec=args.dt_full,
        sgp4_chunk_size=args.sgp4_chunk_size,
        hpop_chunk_size=args.hpop_chunk_size,
        observation_mode=args.observation_mode,
        predict_velocity=args.predict_velocity,
        noise_sigma_r_m=args.noise_sigma_r,
        noise_sigma_t_m=args.noise_sigma_t,
        noise_sigma_n_m=args.noise_sigma_n,
        noise_sigma_vr_mps=args.noise_sigma_vr,
        noise_sigma_vt_mps=args.noise_sigma_vt,
        noise_sigma_vn_mps=args.noise_sigma_vn,
        model_name=args.model_name,
        input_len=args.input_len,
        pred_len=args.pred_len,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        lambda_pred=args.lambda_pred,
        lambda_diff=args.lambda_diff,
        lambda_smooth=args.lambda_smooth,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tcn_kernel_size=args.tcn_kernel_size,
        narx_input_lags=args.narx_input_lags,
        narx_feedback_lags=args.narx_feedback_lags,
        narx_hidden_dim=args.narx_hidden_dim,
        narx_num_hidden_layers=args.narx_num_hidden_layers,
        narx_activation=args.narx_activation,
        narx_dropout=args.narx_dropout,
        narx_snake_alpha=args.narx_snake_alpha,
        narx_prediction_length_sec=args.narx_prediction_length,
        narx_use_velocity_input=args.narx_use_velocity_input,
        optimizer_name=args.optimizer_name,
        optimizer_momentum=args.optimizer_momentum,
        lr_scheduler_name=args.lr_scheduler_name,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        orekit_mass_kg=args.orekit_mass_kg,
        orekit_drag_cd=args.orekit_drag_cd,
        orekit_drag_area_m2=args.orekit_drag_area_m2,
        orekit_srp_cr=args.orekit_srp_cr,
        orekit_srp_area_m2=args.orekit_srp_area_m2,
        orekit_gravity_degree=args.orekit_gravity_degree,
        orekit_gravity_order=args.orekit_gravity_order,
        orekit_enable_third_body=args.orekit_enable_third_body,
        orekit_enable_drag=args.orekit_enable_drag,
        orekit_enable_srp=args.orekit_enable_srp,
        orekit_enable_relativity=args.orekit_enable_relativity,
        orekit_solar_activity_level=args.orekit_solar_activity_level,
        orekit_min_step_sec=args.orekit_min_step_sec,
        orekit_max_step_sec=args.orekit_max_step_sec,
        orekit_abs_tolerance=args.orekit_abs_tolerance,
        orekit_rel_tolerance=args.orekit_rel_tolerance,
        memmap_threshold_mb=args.memmap_threshold_mb,
    )
