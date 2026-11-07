# House Sales Price Prediction

A Python project that cleans a residential house-sales dataset, explores how sale prices vary with the number of bedrooms, and compares a linear regression baseline with a Random Forest regressor.

The project is a learning and portfolio example of a structured workflow: data validation, cleaning with documented assumptions, exploratory summaries, and model comparison in which preprocessing is fitted only on training data.

## What the project does

1. **Data cleaning and validation**
   - Drops records without a usable sale price.
   - Standardizes city and house-type labels against a documented list of allowed values. Unrecognized cities are grouped as `Unknown`.
   - Converts area values written as text (for example `107.8 sq.m.`) to numbers.
   - Imputes missing values with documented assumptions:
     - missing `months_listed`, `bedrooms`, and `area`: the column mean
     - missing `house_type`: the most frequent value
     - missing `sale_date`: a fixed placeholder date (2023-01-01), used only so the column stays complete. This is an assumption, not a measurement.
   - Checks the cleaned table against an expected schema and fails early if it does not match.

2. **Exploratory summary**
   - Mean sale price and sample variance by number of bedrooms.

3. **Modeling**
   - **Baseline:** Linear Regression on the training split.
   - **Comparison:** Random Forest Regressor (100 trees, fixed random seed).
   - Imputers and encoders inside the model pipeline are fitted on training rows only, then applied to validation rows.
   - `house_id` and `sale_date` are excluded from model features.

## Repository structure

```text
.
├── .github/workflows/tests.yml        # runs the test suite on push and pull request
├── src/house_sales_project.py         # cleaning, summary, and modeling pipeline
├── tests/test_house_sales_project.py  # tests on small synthetic data
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

The source data are **not included** in this repository. To run the full pipeline, place your own copy of the files below in `data/raw/`:

- `house_sales.csv`
- `train.csv`
- `validation.csv`

The loader treats `--` as a missing-value marker.

Generated outputs are written to `outputs/` and are excluded from version control.

## Setup and run

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
python src/house_sales_project.py
```

## Tests

The tests use small synthetic data and do not require the raw files:

```bash
python -m pytest -q
```

GitHub Actions runs the same tests on every push and pull request.

## Limitations

- The tests check that the code behaves as designed. They do not show that the predictions are accurate.
- The validation file does not include a reliable sale price in every version of the data, so the script does not report a held-out error metric. Model quality should be assessed with a proper held-out target and metrics (for example RMSE or MAE) before any decision is based on the predictions.
- Cleaning statistics (means and the most frequent house type) are computed on the full `house_sales.csv` file. The model's own imputers (median for numeric features, most frequent value for categorical features) are fitted only on the training split.
- The fixed placeholder date (2023-01-01) and mean imputation are simple choices. They can bias results if missingness is not random.
- This is a learning and portfolio project, not a production pricing system.

## Certification

Data Scientist Associate, DataCamp (credential: DSA0018741766310)

## Acknowledgment

This independent portfolio project was informed by a learning exercise. No course datasets or official course materials are included.
