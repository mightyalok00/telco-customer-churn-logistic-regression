# Data

## Dataset

Expected file:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The repository includes the supplied Telco Customer Churn CSV so a fresh clone can reproduce the analysis without a manual dataset download.

- Rows: **7,043**
- Columns: **21**
- Target: `Churn`
- Identifier: `customerID`

## Data quality

`TotalCharges` is supplied as text and contains blank values. Cleaning converts it to numeric values; invalid/blank entries become missing values and are handled by training-set imputation.

## Redistribution note

The raw CSV is included for reproducibility. Before redistributing the repository or dataset independently, verify the original dataset's licensing and source terms.
