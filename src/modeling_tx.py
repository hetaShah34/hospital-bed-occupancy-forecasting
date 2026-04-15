from __future__ import annotations

import pandas as pd

from modeling_common import (
    default_exog,
    evaluate_candidate_models,
    fit_ets,
    fit_sarimax,
    forecast_with_model,
    naive_forecast,
    plot_forecast,
    prepare_state_series,
    train_test_split_state,
)

STATE = "TX"


def run_tx_modeling(df: pd.DataFrame, plot: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    state_df = prepare_state_series(df, STATE)
    split = train_test_split_state(state_df, test_horizon=13)
    exog_cols = default_exog(state_df)

    baseline_pred = naive_forecast(split.train, split.test)
    ets_fit = fit_ets(split.train, error="add", trend=None, seasonal=None)
    ets_pred = forecast_with_model(ets_fit, steps=len(split.test))
    ets_pred.index = split.test.index

    sarimax_fit = fit_sarimax(split.train, order=(2, 1, 1), seasonal_order=(1, 0, 0, 52), exog_cols=exog_cols)
    sarimax_pred = forecast_with_model(sarimax_fit, steps=len(split.test), exog_future=split.test[exog_cols])
    sarimax_pred.index = split.test.index

    metrics = evaluate_candidate_models(
        split.train,
        split.test,
        {
            "Baseline": baseline_pred,
            "ETS": ets_pred,
            "SARIMAX(2,1,1)(1,0,0,52)": sarimax_pred,
        },
    )

    if plot:
        plot_forecast(split.train, split.test, ets_pred, "Texas Forecast: ETS")

    return metrics, ets_pred
