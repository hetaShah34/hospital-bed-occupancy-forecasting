from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_preprocessing import DATE_COL, REGION_COL, TARGET_COL, COVID_ADM_COL, FLU_ADM_COL
from evaluation import ForecastMetrics, evaluate_forecast, summarize_metrics


@dataclass
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame


def prepare_state_series(df: pd.DataFrame, state: str) -> pd.DataFrame:
    state_df = df[df[REGION_COL] == state].copy()
    state_df = state_df.sort_values(DATE_COL).set_index(DATE_COL)
    if state_df.empty:
        raise ValueError(f"No rows found for state {state}")
    return state_df


def train_test_split_state(df: pd.DataFrame, test_horizon: int = 13) -> SplitData:
    if len(df) <= test_horizon:
        raise ValueError("Not enough observations for the requested test horizon")
    return SplitData(train=df.iloc[:-test_horizon].copy(), test=df.iloc[-test_horizon:].copy())


def naive_forecast(train: pd.DataFrame, test: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    return pd.Series([train[target_col].iloc[-1]] * len(test), index=test.index, name="baseline")


def fit_ets(
    train: pd.DataFrame,
    error: str = "add",
    trend: str | None = None,
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    target_col: str = TARGET_COL,
):
    model = ETSModel(
        train[target_col],
        error=error,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    return model.fit(disp=False)


def fit_sarimax(
    train: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    exog_cols: Iterable[str] | None = None,
    target_col: str = TARGET_COL,
):
    exog = train[list(exog_cols)] if exog_cols else None
    model = SARIMAX(
        train[target_col],
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_with_model(result, steps: int, exog_future: pd.DataFrame | None = None) -> pd.Series:
    forecast = result.get_forecast(steps=steps, exog=exog_future)
    return forecast.predicted_mean


def evaluate_candidate_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_predictions: dict[str, pd.Series],
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    metrics = [evaluate_forecast(name, test[target_col], pred) for name, pred in candidate_predictions.items()]
    return summarize_metrics(metrics)


def plot_forecast(train: pd.DataFrame, test: pd.DataFrame, forecast: pd.Series, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train.index, train[TARGET_COL], label="Train")
    ax.plot(test.index, test[TARGET_COL], label="Test")
    ax.plot(forecast.index, forecast.values, label="Forecast")
    ax.set_title(title)
    ax.set_xlabel(DATE_COL)
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def default_exog(df: pd.DataFrame) -> list[str]:
    cols = [c for c in [COVID_ADM_COL, FLU_ADM_COL] if c in df.columns]
    return cols
