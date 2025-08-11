# ===========================================
# 1. Import Libraries
# ===========================================
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# ===========================================
# 2. Load Datasets
# ===========================================
baseline = pd.read_csv("Baseline_MentalHealth_with_BEEPS.csv")
enhanced = pd.read_csv("Enhanced_MentalHealth_with_BEEPS.csv")
realworld = pd.read_csv("RealWorld_MentalHealth_DiD_Dataset.csv")

# Initialise metadata trackers for variable source/method
def init_metadata(df):
    return {col: "original_source" for col in df.columns}

baseline_meta = init_metadata(baseline)
enhanced_meta = init_metadata(enhanced)
realworld_meta = init_metadata(realworld)

# ===========================================
# 3. Data Transformations
# ===========================================

# Standardisation of monetary units
cso_deflator_2022 = 1.05  # Example, replace with actual
for df, meta in zip([baseline, enhanced, realworld],
                    [baseline_meta, enhanced_meta, realworld_meta]):
    for col in ['turnover_d2', 'revenue_per_employee_d2']:
        if col in df.columns:
            df[col] = df[col] / cso_deflator_2022
            meta[col] = "transformation_standardisation"

# Likert to binary
likert_vars = ['mh_agree_strong', 'leadership_support']
for df, meta in zip([baseline, enhanced, realworld],
                    [baseline_meta, enhanced_meta, realworld_meta]):
    for var in likert_vars:
        if var in df.columns:
            df[var] = np.where(df[var] >= 4, 1, 0)
            meta[var] = "transformation_likert_to_binary"

# Sector recoding
sector_map = {
    'Information & Communication': 'ICT/Finance',
    'Financial Services': 'ICT/Finance',
    'Healthcare': 'Healthcare/Education',
    'Education': 'Healthcare/Education',
    'Retail': 'Retail/Hospitality',
    'Hospitality': 'Retail/Hospitality',
    'Manufacturing': 'Manufacturing',
    'Construction': 'Construction/Logistics',
    'Logistics': 'Construction/Logistics'
}
for df, meta in zip([baseline, enhanced, realworld],
                    [baseline_meta, enhanced_meta, realworld_meta]):
    if 'main_sector' in df.columns:
        df['sector'] = df['main_sector'].map(sector_map)
        meta['sector'] = "derived_sector_recoding"

# Log transformation
for df, meta in zip([baseline, enhanced, realworld],
                    [baseline_meta, enhanced_meta, realworld_meta]):
    for col in ['revenue_per_employee_d2', 'turnover_d2']:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            meta[col] = "transformation_log"

# ===========================================
# 4. Imputation Logic
# ===========================================
def sector_size_median_impute(df, col, meta):
    if col in df.columns:
        df[col] = df.groupby(['sector', 'firm_size'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        meta[col] = "imputation_sector_size_median"

for col in ['absenteeism_days', 'turnover_rate']:
    for df, meta in zip([baseline, enhanced, realworld],
                        [baseline_meta, enhanced_meta, realworld_meta]):
        sector_size_median_impute(df, col, meta)

def hot_deck_impute(df, col, meta):
    if col in df.columns:
        for size in df['firm_size'].unique():
            for sec in df['sector'].unique():
                mask = (df['firm_size'] == size) & (df['sector'] == sec)
                mode_val = df.loc[mask, col].mode()
                if not mode_val.empty:
                    df.loc[mask & df[col].isna(), col] = mode_val[0]
        meta[col] = "imputation_hot_deck"

for col in ['leadership_support']:
    for df, meta in zip([baseline, enhanced, realworld],
                        [baseline_meta, enhanced_meta, realworld_meta]):
        hot_deck_impute(df, col, meta)

def regression_impute(df, target_col, predictor_cols, meta):
    if target_col in df.columns and all(col in df.columns for col in predictor_cols):
        missing_mask = df[target_col].isna()
        train_df = df[~missing_mask].dropna(subset=predictor_cols)
        test_df = df[missing_mask]
        if not train_df.empty and not test_df.empty:
            model = LinearRegression()
            model.fit(train_df[predictor_cols], train_df[target_col])
            df.loc[missing_mask, target_col] = model.predict(test_df[predictor_cols])
            meta[target_col] = "imputation_regression"

for df, meta in zip([baseline, enhanced, realworld],
                    [baseline_meta, enhanced_meta, realworld_meta]):
    regression_impute(df, 'manager_age', ['manager_education', 'firm_size_code', 'region_code'], meta)

# ===========================================
# 5. Validation Checks
# ===========================================
for df_name, df in zip(['Baseline', 'Enhanced', 'RealWorld'], [baseline, enhanced, realworld]):
    if 'absenteeism_days' in df.columns:
        observed = df['absenteeism_days'].dropna()
        imputed = df['absenteeism_days'][df['absenteeism_days'].isna()]
        if not imputed.empty and not observed.empty:
            stat, pval = ks_2samp(observed, imputed)
            print(f"{df_name} KS-Test p-value for absenteeism_days: {pval}")

# Outlier detection
for df_name, df in zip(['Baseline', 'Enhanced', 'RealWorld'], [baseline, enhanced, realworld]):
    if 'absenteeism_days' in df.columns:
        mean_val, std_val = df['absenteeism_days'].mean(), df['absenteeism_days'].std()
        outlier_count = ((df['absenteeism_days'] < mean_val - 3 * std_val) |
                         (df['absenteeism_days'] > mean_val + 3 * std_val)).sum()
        print(f"{df_name} absenteeism outliers beyond ±3 SD: {outlier_count}")

# ===========================================
# 6. Multicollinearity Diagnosis
# ===========================================
def compute_vif(df, vars_list):
    X = df[vars_list].dropna()
    X = sm.add_constant(X)
    return pd.DataFrame({
        'Variable': vars_list,
        'VIF': [variance_inflation_factor(X.values, i+1) for i in range(len(vars_list))]
    })

if all(col in realworld.columns for col in ['treat', 'post', 'treat_post']):
    print("Original RealWorld VIFs:")
    print(compute_vif(realworld, ['treat', 'post', 'treat_post']))

# ===========================================
# 7. Cleaning to Remove Multicollinearity
# ===========================================
realworld_clean = realworld.drop(columns=['treat_post'], errors='ignore')
realworld_clean_meta = {k: v for k, v in realworld_meta.items() if k in realworld_clean.columns}

if all(col in realworld_clean.columns for col in ['treat', 'post']):
    print("Cleaned RealWorld VIFs:")
    print(compute_vif(realworld_clean, ['treat', 'post']))

# ===========================================
# 8. Append Metadata to Datasets
# ===========================================
def append_metadata(df, meta_dict):
    meta_df = pd.DataFrame(list(meta_dict.items()), columns=['Variable', 'source_method'])
    return df, meta_df

baseline, baseline_meta_df = append_metadata(baseline, baseline_meta)
enhanced, enhanced_meta_df = append_metadata(enhanced, enhanced_meta)
realworld_clean, realworld_meta_df = append_metadata(realworld_clean, realworld_clean_meta)

# ===========================================
# 9. Save Outputs
# ===========================================
baseline.to_csv("Baseline_Policy_Clean.csv", index=False)
enhanced.to_csv("Enhanced_Policy_Clean.csv", index=False)
realworld_clean.to_csv("RealWorld_DiD_Clean.csv", index=False)

baseline_meta_df.to_csv("Baseline_Policy_Metadata.csv", index=False)
enhanced_meta_df.to_csv("Enhanced_Policy_Metadata.csv", index=False)
realworld_meta_df.to_csv("RealWorld_DiD_Metadata.csv", index=False)

print("Cleaned datasets and metadata files saved successfully.")
