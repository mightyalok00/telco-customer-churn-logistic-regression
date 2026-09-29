# Data

The project uses the supplied Telco Customer Churn CSV.

Expected file:
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The source file contains 7,043 rows and 21 columns. The target is `Churn`.

`TotalCharges` is supplied as text and contains blank values that become missing after numeric coercion. The modeling pipeline handles this safely through imputation learned from training data.

For redistribution/licensing questions, verify the dataset's original Kaggle/source terms before publishing the raw CSV publicly.
