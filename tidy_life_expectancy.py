"""
Tidying the WHO Life Expectancy dataset (Kaggle, Rajarshi 2018).
Pipeline: inspect -> standardise -> validate -> repair/flag -> impute -> evaluate.
Run:  python tidy_life_expectancy.py  (with "Life Expectancy Data.csv" in the same folder)
"""
import json
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RAW = "Life Expectancy Data.csv"
log = {}

# ---------------------------------------------------------------- 1 load
raw = pd.read_csv(RAW)
log["raw_shape"] = raw.shape
log["cols_with_whitespace"] = int(sum(c != c.strip() for c in raw.columns))

# ---------------------------------------------------------------- 2 standardise names
rename = {
    "Country": "country", "Year": "year", "Status": "status",
    "Life expectancy ": "life_expectancy", "Adult Mortality": "adult_mortality",
    "infant deaths": "infant_deaths", "Alcohol": "alcohol",
    "percentage expenditure": "pct_expenditure", "Hepatitis B": "hepatitis_b",
    "Measles ": "measles", " BMI ": "bmi", "under-five deaths ": "under_five_deaths",
    "Polio": "polio", "Total expenditure": "total_expenditure",
    "Diphtheria ": "diphtheria", " HIV/AIDS": "hiv_aids", "GDP": "gdp",
    "Population": "population",
    # WHO publishes thinness for 5-9, 10-19 and 5-19 years; no 1-19 band exists
    " thinness  1-19 years": "thinness_10_19", " thinness 5-9 years": "thinness_5_9",
    "Income composition of resources": "income_composition", "Schooling": "schooling",
}
df = raw.rename(columns=rename).sort_values(["country", "year"]).reset_index(drop=True)
log["duplicate_country_years"] = int(df.duplicated(["country", "year"]).sum())
log["missing_raw"] = df.isna().sum().to_dict()
log["complete_rows_raw"] = int(df.dropna().shape[0])

# ---------------------------------------------------------------- 3 validate
issues = {}
# 3a unbalanced panel: countries with a single year and no target
counts = df.groupby("country").size()
single = counts[counts == 1].index.tolist()
issues["single_year_countries"] = single
# 3b disguised missing values (zeros that are impossible)
issues["zero_income_composition"] = int((df.income_composition == 0).sum())
issues["zero_schooling"] = int((df.schooling == 0).sum())
issues["zero_pct_expenditure"] = int((df.pct_expenditure == 0).sum())
# 3c out-of-range values (national mean adult BMI plausibly 15-40 kg/m2)
issues["bmi_out_of_range"] = int(((df.bmi < 15) | (df.bmi > 40)).sum())
issues["bmi_non_missing"] = int(df.bmi.notna().sum())
# 3d counts labelled as rates: deaths "per 1000" above 1000
issues["infant_deaths_gt_1000"] = int((df.infant_deaths > 1000).sum())
issues["under5_gt_1000"] = int((df.under_five_deaths > 1000).sum())
# 3e countries whose GDP is missing in every year (failed name match)
allmiss = df.groupby("country").gdp.apply(lambda s: s.isna().all())
issues["countries_gdp_all_missing"] = int(allmiss.sum())
issues["countries_gdp_all_missing_list"] = allmiss[allmiss].index.tolist()


# 3f within-country jumps: compare each value with the median of its
#    neighbours (+-2 years, excluding itself) inside the same country
def local_jump(s, log_scale=False, k=2):
    v = np.log10(s.where(s > 0)) if log_scale else s
    out = pd.Series(np.nan, index=s.index)
    arr = v.to_numpy()
    for i in range(len(arr)):
        nb = np.r_[arr[max(0, i - k):i], arr[i + 1:i + 1 + k]]
        nb = nb[~np.isnan(nb)]
        if len(nb) >= 2 and not np.isnan(arr[i]):
            out.iloc[i] = arr[i] - np.median(nb)
    return out


def flag(col, thr, log_scale=False, downward_only=False):
    d = df.groupby("country", group_keys=False)[col].apply(
        lambda s: local_jump(s, log_scale))
    # truncated digits can only make a value smaller -> optional one-sided rule
    return (d < -thr) if downward_only else (d.abs() > thr)


flags = {
    "life_expectancy": flag("life_expectancy", 5),         # >5 years jump
    "adult_mortality": flag("adult_mortality", 0.5, True, True),  # >~3x drop
    "gdp": flag("gdp", 0.7, True),                         # >~5x jump
    "population": flag("population", 0.7, True),
}
issues["jump_flags"] = {k: int(v.sum()) for k, v in flags.items()}
# 3g status label vs. life expectancy: "Developing" countries with LE >= 80
le_mean = df.groupby("country").agg(status=("status", "first"),
                                    mean_le=("life_expectancy", "mean"))
issues["developing_with_mean_le_ge_80"] = le_mean[(le_mean.status == "Developing")
                                                  & (le_mean.mean_le >= 80)].index.tolist()

# 3h year-on-year jumps larger than factor 5 (implausible for these indicators)
def yoy_share(col):
    d = df.groupby("country")[col].apply(lambda s: np.log10(s.where(s > 0)).diff().abs())
    return round(float((d > 0.7).sum() / d.notna().sum()), 3)
issues["yoy_jump_gt5x_share"] = {c: yoy_share(c) for c in ["gdp", "population"]}
log["issues"] = issues

# ---------------------------------------------------------------- 4 repair
clean = df.copy()
clean = clean[~clean.country.isin(single)]                     # unbalanced panel
for c in ["income_composition", "schooling"]:                  # disguised missing
    clean.loc[clean[c] == 0, c] = np.nan
for c in ["life_expectancy", "adult_mortality"]:               # isolated jumps
    clean.loc[flags[c].reindex(clean.index).fillna(False), c] = np.nan
# variables that cannot be repaired without an external source -> drop
dropped = {
    "population": "factor-10+ jumps in many country-years; needs external source",
    "gdp": "factor-10 errors occur in runs; local rule cannot repair them",
    "pct_expenditure": "611 zeros; derived from the corrupted GDP column",
    "bmi": "two thirds of values outside 15-40 kg/m2; definition unclear",
    "infant_deaths": "absolute counts labelled as rates; no valid denominator",
    "under_five_deaths": "absolute counts labelled as rates; no valid denominator",
    "measles": "absolute counts; no valid denominator",
    "thinness_5_9": "near-duplicate of thinness_10_19",
    "status": "time-invariant label; misclassifies e.g. Canada, France",
}
clean = clean.drop(columns=list(dropped))
log["dropped"] = dropped
log["corr_thinness"] = round(df.thinness_10_19.corr(df.thinness_5_9), 3)
clean["log_hiv_aids"] = np.log10(clean.pop("hiv_aids"))        # right skew -> log

# 4b impute: (i) linear interpolation within country, (ii) iterative imputer
num = [c for c in clean.columns if c not in ("country", "year")]
was_missing = clean[num].isna()
log["missing_after_flags"] = was_missing.sum().to_dict()
interp = clean.copy()
interp[num] = interp.groupby("country")[num].transform(
    lambda s: s.interpolate(limit_direction="both"))
log["missing_after_interp"] = interp[num].isna().sum().to_dict()
pred = [c for c in num if c != "life_expectancy"]  # target left out of imputation
imp = IterativeImputer(max_iter=20, random_state=42)
interp[pred] = imp.fit_transform(interp[pred])
tidy = interp
tidy["le_imputed"] = was_missing["life_expectancy"]            # keep provenance
tidy["n_imputed"] = was_missing.drop(columns="life_expectancy").sum(axis=1)
log["share_cells_imputed"] = round(float(was_missing.to_numpy().mean()), 3)
log["tidy_shape"] = tidy.shape
tidy.to_csv("life_expectancy_tidy.csv", index=False)
log["corr_income_schooling"] = round(tidy.income_composition.corr(tidy.schooling), 3)

# ---------------------------------------------------------------- 5 evaluate
def evaluate(data, features, label):
    d = data.dropna(subset=features + ["life_expectancy"])
    X, y, g = d[features], d.life_expectancy, d.country
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
    coefs = dict(zip(features, lin[-1].coef_.round(2)))
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    r2_rand = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=42),
                              scoring="r2").mean()
    r2_group = cross_val_score(rf, X, y, cv=GroupKFold(5), groups=g,
                               scoring="r2").mean()
    lin_group = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()),
                                X, y, cv=GroupKFold(5), groups=g, scoring="r2").mean()
    return {"label": label, "n_rows": len(d), "n_countries": int(g.nunique()),
            "ols_r2_in_sample": round(lin.score(X, y), 3),
            "ols_r2_grouped_cv": round(lin_group, 3),
            "rf_r2_random_cv": round(r2_rand, 3),
            "rf_r2_grouped_cv": round(r2_group, 3), "std_coefs": coefs}


raw_feats = ["adult_mortality", "alcohol", "hepatitis_b", "bmi", "polio",
             "total_expenditure", "diphtheria", "hiv_aids", "gdp", "population",
             "thinness_10_19", "income_composition", "schooling"]
tidy_feats = ["adult_mortality", "alcohol", "hepatitis_b", "polio",
              "total_expenditure", "diphtheria", "log_hiv_aids",
              "thinness_10_19", "income_composition", "schooling"]
log["eval_raw_listwise"] = evaluate(df, raw_feats, "A raw, listwise deletion")
log["eval_flagged_listwise"] = evaluate(clean, tidy_feats, "B validated, listwise deletion")
# observed target only: imputed life expectancy is never used as ground truth
log["eval_tidy"] = evaluate(tidy[~tidy.le_imputed], tidy_feats, "C validated + imputed")
log["raw_corr_bmi_le"] = round(df.bmi.corr(df.life_expectancy), 3)

with open("tidy_log.json", "w") as f:
    json.dump(log, f, indent=1, default=str)
print(json.dumps({k: v for k, v in log.items() if k not in ("missing_raw",)}, indent=1, default=str))
