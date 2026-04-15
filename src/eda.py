from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from data_preprocessing import DATE_COL, REGION_COL, TARGET_COL, COVID_ADM_COL, FLU_ADM_COL

sns.set_theme(style="whitegrid")


def get_top_regions(df: pd.DataFrame, k: int = 3) -> list[str]:
    return (
        df.groupby(REGION_COL)[TARGET_COL]
        .mean()
        .sort_values(ascending=False)
        .head(k)
        .index.tolist()
    )


def plot_national_occupancy(df: pd.DataFrame, out_path: str | Path | None = None) -> None:
    overall = df.groupby(DATE_COL)[TARGET_COL].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(overall[DATE_COL], overall[TARGET_COL])
    ax.set_title("Total Inpatient Beds Occupied (All Regions Combined)")
    ax.set_xlabel(DATE_COL)
    ax.set_ylabel(TARGET_COL)
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, out_path)


def plot_region_rankings(df: pd.DataFrame, out_path: str | Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    region_avg = df.groupby(REGION_COL)[TARGET_COL].mean().sort_values(ascending=False)
    axes[0].bar(region_avg.index, region_avg.values)
    axes[0].set_title("Regions by Average Inpatient Bed Occupancy")
    axes[0].tick_params(axis="x", rotation=90)
    axes[1].hist(df[TARGET_COL].dropna(), bins=30)
    axes[1].set_title("Distribution of Weekly Inpatient Bed Occupancy")
    _save_or_show(fig, out_path)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: str | Path | None = None) -> pd.DataFrame:
    corr_vars = [TARGET_COL, COVID_ADM_COL, FLU_ADM_COL]
    corr = df[corr_vars].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    _save_or_show(fig, out_path)
    return corr


def plot_top_region_series(df: pd.DataFrame, regions: Sequence[str] | None = None, out_path: str | Path | None = None) -> None:
    regions = list(regions or get_top_regions(df, 3))
    fig, ax = plt.subplots(figsize=(14, 7))
    subset = df[df[REGION_COL].isin(regions)]
    sns.lineplot(data=subset, x=DATE_COL, y=TARGET_COL, hue=REGION_COL, ax=ax)
    ax.set_title("Top Regions: Inpatient Beds Occupied Over Time")
    _save_or_show(fig, out_path)


def decompose_series(df: pd.DataFrame, region: str | None = None, out_dir: str | Path | None = None) -> None:
    work = df.copy()
    if region is not None:
        work = work[work[REGION_COL] == region]
        title = f"{region} decomposition"
    else:
        work = work.groupby(DATE_COL)[TARGET_COL].sum().reset_index()
        title = "National decomposition"
    work = work.sort_values(DATE_COL).set_index(DATE_COL)
    result = seasonal_decompose(work[TARGET_COL], model="additive", period=52)
    fig = result.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle(title)
    _save_or_show(fig, _resolve_dir_path(out_dir, f"decomposition_{region or 'national'}.png"))


def stationarity_report(df: pd.DataFrame, regions: Sequence[str] | None = None, max_lag: int = 60) -> pd.DataFrame:
    regions = list(regions or get_top_regions(df, 3))
    rows: list[dict[str, float | str]] = []
    for region in regions:
        series = (
            df[df[REGION_COL] == region]
            .sort_values(DATE_COL)[TARGET_COL]
            .dropna()
        )
        stat, p_value, used_lag, _, _, _ = adfuller(series)
        rows.append({
            "region": region,
            "adf_statistic": stat,
            "p_value": p_value,
            "used_lag": used_lag,
        })

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(series, lags=max_lag, ax=axes[0])
        plot_pacf(series, lags=max_lag, ax=axes[1], method="ywm")
        axes[0].set_title(f"{region} ACF")
        axes[1].set_title(f"{region} PACF")
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(rows)


def plot_lag_relationships(df: pd.DataFrame, region: str, lags: Iterable[int] = (1, 2, 4, 8, 26, 52)) -> None:
    series = (
        df[df[REGION_COL] == region]
        .sort_values(DATE_COL)[TARGET_COL]
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, lag in zip(axes.flatten(), lags):
        lag_plot(series, lag=lag, ax=ax)
        ax.set_title(f"{region}: lag {lag}")
    plt.tight_layout()
    plt.show()


def _resolve_dir_path(out_dir: str | Path | None, filename: str) -> str | Path | None:
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def _save_or_show(fig: plt.Figure, out_path: str | Path | None) -> None:
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
