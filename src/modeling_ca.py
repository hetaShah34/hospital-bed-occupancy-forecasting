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

STATE = "CA"


def run_ca_modeling(df: pd.DataFrame, plot: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    state_df = prepare_state_series(df, STATE)
    split = train_test_split_state(state_df, test_horizon=13)
    exog_cols = default_exog(state_df)

    baseline_pred = naive_forecast(split.train, split.test)
    arima_fit = fit_sarimax(split.train, order=(2, 0, 0))
    arima_pred = forecast_with_model(arima_fit, steps=len(split.test))
    arima_pred.index = split.test.index

    sarima_fit = fit_sarimax(split.train, order=(2, 0, 0), seasonal_order=(1, 0, 0, 52))
    sarima_pred = forecast_with_model(sarima_fit, steps=len(split.test))
    sarima_pred.index = split.test.index

    sarimax_fit = fit_sarimax(split.train, order=(2, 0, 0), seasonal_order=(1, 0, 0, 52), exog_cols=exog_cols)
    sarimax_pred = forecast_with_model(sarimax_fit, steps=len(split.test), exog_future=split.test[exog_cols])
    sarimax_pred.index = split.test.index

    ets_fit = fit_ets(split.train, error="add", trend=None, seasonal="add", seasonal_periods=52)
    ets_pred = forecast_with_model(ets_fit, steps=len(split.test))
    ets_pred.index = split.test.index

    metrics = evaluate_candidate_models(
        split.train,
        split.test,
        {
            "Baseline": baseline_pred,
            "ARIMA(2,0,0)": arima_pred,
            "SARIMA(2,0,0)(1,0,0,52)": sarima_pred,
            "SARIMAX(2,0,0)(1,0,0,52)": sarimax_pred,
            "Seasonal ETS": ets_pred,
        },
    )

    if plot:
        plot_forecast(split.train, split.test, sarimax_pred, "California Forecast: SARIMAX")

    return metrics, sarimax_pred
