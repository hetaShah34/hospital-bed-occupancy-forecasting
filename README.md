# Hospital Bed Occupancy Forecasting

This repository contains a cleaned, modular version of a group predictive analytics project on forecasting weekly hospital inpatient bed occupancy.

## Recommended repo structure

```text
hospital-bed-occupancy-forecasting/
├── README.md
├── requirements.txt
├── data/
│   └── processed/
├── docs/
│   ├── final_presentation.pdf
│   └── final_report.docx
├── notebooks/
│   └── hospital_bed_occupancy_workflow.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
└── src/
    ├── data_preprocessing.py
    ├── eda.py
    ├── evaluation.py
    ├── modeling_common.py
    ├── modeling_tx.py
    ├── modeling_ca.py
    └── modeling_fl.py
```

## What each file does

- `src/data_preprocessing.py`: loads the raw dataset, standardizes dates, removes unreliable periods, fixes the Georgia outlier, interpolates missing values, and saves the processed dataset.
- `src/eda.py`: reusable functions for national trend plots, regional rankings, correlation heatmaps, decomposition, and stationarity diagnostics.
- `src/modeling_tx.py`: modeling workflow for Texas.
- `src/modeling_ca.py`: modeling workflow for California.
- `src/modeling_fl.py`: modeling workflow for Florida.
- `src/evaluation.py`: RMSE, MAE, MAPE, and metrics table helpers.
- `notebooks/hospital_bed_occupancy_workflow.ipynb`: cleaned notebook that ties the pipeline together.

```

## Suggested next step

Add raw dataset locally (at https://www.kaggle.com/datasets/noeyislearning/weekly-hospital-respiratory-data-and-metrics/data),
then run the notebook or the preprocessing script first.
