# Tidying the WHO Life Expectancy Dataset



The script finds and treats data quality problems in the Kaggle *Life Expectancy (WHO)* dataset and measures how the treatment changes the results of a prediction model.

## What the script does

1. **Standardise names**: removes stray spaces from column names and fixes the mislabelled thinness column.
2. **Validate**: checks for single-year countries, zeros that hide missing values, impossible BMI values, counts labelled as rates, and implausible year-to-year jumps (compared with the median of each country's neighbouring years).
3. **Repair**: sets invalid values to missing and drops variables that cannot be repaired from the file itself (population, GDP, BMI, death counts, etc.).
4. **Impute**: fills gaps by linear interpolation within each country (nearest value at series ends), then with scikit-learn's `IterativeImputer`. Life expectancy is left out of the imputation model.
5. **Evaluate**: compares linear regression and random forest models on three versions of the data (raw, validated, validated + imputed), using cross-validation grouped by country.

## How to run

1. Install Python 3.10 or newer, then install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Download the raw data from Kaggle: [Life Expectancy (WHO) by KumarRajarshi](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who). Unzip it and put `Life Expectancy Data.csv` in this folder. The raw file is not included in this repository; please get it from the original source.
3. Run:
   ```
   python tidy_life_expectancy.py
   python make_figures.py
   ```

## Output

- `life_expectancy_tidy.csv`: the cleaned dataset (2,928 rows, 183 countries)
- `tidy_log.json`: the main statistics (issue counts, missing values) and all model results reported in the assignment
- `figure1_missing_values.png` / `.svg`: Figure 1 of the assignment (missing values per variable)

Random seeds are fixed, so the results are reproducible. They were produced with Python 3.13, pandas 3.0, NumPy 2.5 and scikit-learn 1.9.

## Data source

Rajarshi, K. (2018). *Life expectancy (WHO)* [Data set]. Kaggle. https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
