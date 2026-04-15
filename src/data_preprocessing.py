from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DATE_COL = "Week Ending Date"
REGION_COL = "Geographic aggregation"
TARGET_COL = "Number of Inpatient Beds Occupied"
BEDS_COL = "Number of Inpatient Beds"
COVID_ADM_COL = "Total COVID-19 Admissions"
FLU_ADM_COL = "Total Influenza Admissions"
RSV_ADM_COL = "Total RSV Admissions"
REPORTING_BED_OCC_COL = "Percent Hospitals Reporting Number of Inpatient Beds Occupied"

MANDATORY_END_DATE = pd.Timestamp("2024-05-04")
GA_OUTLIER_DATE = pd.Timestamp("2021-01-23")
GA_REGION = "GA"

DEFAULT_KEEP_COLUMNS = [
    DATE_COL,
    REGION_COL,
    BEDS_COL,
    TARGET_COL,
    COVID_ADM_COL,
    FLU_ADM_COL,
    RSV_ADM_COL,
    "Percent Hospitals Reporting Total Patients Hospitalized with COVID-19",
    "Percent Hospitals Reporting Total Patients Hospitalized with Influenza",
    "Percent Hospitals Reporting Total Patients Hospitalized with RSV",
    "Percent Hospitals Reporting COVID-19 Admissions",
    "Percent Hospitals Reporting Influenza Admissions",
    "Percent Hospitals Reporting RSV Admissions",
    REPORTING_BED_OCC_COL,
]

PERCENT_COLUMNS = [
    "Percent Hospitals Reporting Total Patients Hospitalized with COVID-19",
    "Percent Hospitals Reporting Total Patients Hospitalized with Influenza",
    "Percent Hospitals Reporting Total Patients Hospitalized with RSV",
    "Percent Hospitals Reporting COVID-19 Admissions",
    "Percent Hospitals Reporting Influenza Admissions",
    "Percent Hospitals Reporting RSV Admissions",
    REPORTING_BED_OCC_COL,
]


@dataclass
class PreprocessSummary:
    rows_before: int
    rows_after: int
    regions_after: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


class DataPreprocessor:
    """Prepare the weekly hospital respiratory dataset for forecasting."""

    def __init__(self, reporting_threshold: float = 0.80):
        self.reporting_threshold = reporting_threshold

    def load_data(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError("Supported file types: csv, parquet, xlsx, xls")

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessSummary]:
        rows_before = len(df)
        df = df.copy()

        df = self._standardize_dates(df)
        df = self._keep_relevant_columns(df)
        df = self._remove_national_records(df)
        df = self._remove_low_history_regions(df)
        df = self._scale_percent_columns(df)
        df = self._drop_low_reporting_rsv(df)
        df = self._fix_ga_outlier(df)
        df = self._drop_missing_target(df)
        df = self._interpolate_by_region(df)
        df = self._trim_to_mandatory_reporting_period(df)
        df = df.sort_values([REGION_COL, DATE_COL]).reset_index(drop=True)

        summary = PreprocessSummary(
            rows_before=rows_before,
            rows_after=len(df),
            regions_after=df[REGION_COL].nunique(),
            start_date=df[DATE_COL].min(),
            end_date=df[DATE_COL].max(),
        )
        return df, summary

    def save_processed_data(self, df: pd.DataFrame, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            df.to_csv(out_path, index=False)
        elif out_path.suffix.lower() in {".parquet", ".pq"}:
            df.to_parquet(out_path, index=False)
        else:
            raise ValueError("Output file must end in .csv or .parquet")

    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        return df

    def _keep_relevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = [col for col in DEFAULT_KEEP_COLUMNS if col in df.columns]
        return df[keep].copy()

    def _remove_national_records(self, df: pd.DataFrame) -> pd.DataFrame:
        if REGION_COL in df.columns:
            df = df[df[REGION_COL].ne("USA")].copy()
        return df

    def _remove_low_history_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        excluded = {"AS", "MP"}
        return df[~df[REGION_COL].isin(excluded)].copy()

    def _scale_percent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in PERCENT_COLUMNS:
            if col in df.columns:
                max_val = df[col].dropna().max()
                if pd.notna(max_val) and max_val > 1:
                    df[col] = df[col] / 100.0
        return df

    def _drop_low_reporting_rsv(self, df: pd.DataFrame) -> pd.DataFrame:
        rsv_reporting_cols = [
            col for col in [
                RSV_ADM_COL,
                "Percent Hospitals Reporting Total Patients Hospitalized with RSV",
                "Percent Hospitals Reporting RSV Admissions",
            ]
            if col in df.columns
        ]
        if not rsv_reporting_cols:
            return df

        reporting_col = "Percent Hospitals Reporting RSV Admissions"
        if reporting_col in df.columns:
            mean_reporting = df[reporting_col].mean(skipna=True)
            if pd.notna(mean_reporting) and mean_reporting < 0.05:
                df = df.drop(columns=rsv_reporting_cols)
        return df

    def _fix_ga_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df[REGION_COL] == GA_REGION) & (df[DATE_COL] == GA_OUTLIER_DATE)
        if mask.any():
            df.loc[mask, TARGET_COL] = np.nan
        return df

    def _drop_missing_target(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[TARGET_COL].notna()].copy()

    def _interpolate_by_region(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df[col] = (
                df.sort_values([REGION_COL, DATE_COL])
                .groupby(REGION_COL)[col]
                .transform(lambda s: s.interpolate(method="linear").ffill().bfill())
            )
        return df

    def _trim_to_mandatory_reporting_period(self, df: pd.DataFrame) -> pd.DataFrame:
        if REPORTING_BED_OCC_COL in df.columns:
            weekly_reporting = (
                df.groupby(DATE_COL)[REPORTING_BED_OCC_COL]
                .mean()
                .sort_index()
            )
            low_weeks = weekly_reporting[weekly_reporting < self.reporting_threshold]
            if not low_weeks.empty:
                first_low_week = low_weeks.index.min()
                cutoff = min(first_low_week - pd.Timedelta(days=7), MANDATORY_END_DATE)
                return df[df[DATE_COL] <= cutoff].copy()
        return df[df[DATE_COL] <= MANDATORY_END_DATE].copy()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["quarter"] = df[DATE_COL].dt.quarter
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    lags: Iterable[int],
    region_col: str = REGION_COL,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    df = df.copy().sort_values([region_col, date_col])
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df.groupby(region_col)[col].shift(lag)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess hospital bed occupancy data")
    parser.add_argument("input_path", help="Path to the raw data file")
    parser.add_argument("output_path", help="Path to save the processed data (.csv or .parquet)")
    args = parser.parse_args()

    processor = DataPreprocessor()
    raw = processor.load_data(args.input_path)
    clean, summary = processor.preprocess(raw)
    processor.save_processed_data(clean, args.output_path)

    print("Saved processed dataset")
    print(summary)
