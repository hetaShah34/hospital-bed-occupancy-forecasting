from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class ForecastMetrics:
    model_name: str
    rmse: float
    mae: float
    mape: float


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_forecast(model_name: str, y_true: pd.Series, y_pred: pd.Series) -> ForecastMetrics:
    return ForecastMetrics(
        model_name=model_name,
        rmse=rmse(y_true, y_pred),
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=mape(y_true, y_pred),
    )


def summarize_metrics(metrics: Iterable[ForecastMetrics]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model": m.model_name,
            "rmse": m.rmse,
            "mae": m.mae,
            "mape": m.mape,
        }
        for m in metrics
    ]).sort_values("rmse")


def save_metrics_table(metrics_df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path, index=False)
