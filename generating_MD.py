import numpy as np
import pandas as pd
import json
import os

def load_config():
    with open('configuration.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# --- 1. Environment setup and configuration loading ---
config = load_config()
cfg_data = config['dataset_config']

# Construct relative paths
input_path = os.path.join(".", "data", cfg_data['input_raw_name'])
output_path = os.path.join(".", "data", cfg_data['output_md_name'])

if not os.path.exists(input_path):
    raise FileNotFoundError(f"File not found in ./data/ directory: {cfg_data['input_raw_name']}")

# --- 2. Load and validate data format ---
df = pd.read_csv(input_path)
print(f"Data loaded successfully. Initial sample count: {len(df)}")

# Validate required column names
required_columns = ['sample_id', 'dataset_type']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Input dataset is missing required column: {col}")

# Validate data ordering (calibration first, then test)
type_series = df['dataset_type'].values
first_test_idx = np.where(type_series == 'test')[0]

if len(first_test_idx) > 0:
    idx = first_test_idx[0]
    if 'calibration' in type_series[idx:]:
        raise ValueError("Dataset format error: interleaving detected. All calibration samples must come before test samples.")

# --- 3. Truncate and split data based on days ---
samples_per_day = cfg_data['samples_per_day']

df_calib = df[df['dataset_type'] == 'calibration'].copy()
df_test = df[df['dataset_type'] == 'test'].copy()

# Compute number of full days and discard remainder
calib_days = len(df_calib) // samples_per_day
test_days = len(df_test) // samples_per_day

df_calib_cleaned = df_calib.iloc[:calib_days * samples_per_day]
df_test_cleaned = df_test.iloc[:test_days * samples_per_day]

print(f"Calibration set: {len(df_calib_cleaned)} samples -> {calib_days} days (discarded {len(df_calib) % samples_per_day} samples)")
print(f"Test set: {len(df_test_cleaned)} samples -> {test_days} days (discarded {len(df_test) % samples_per_day} samples)")

# Merge cleaned feature data (excluding non-feature columns)
# Feature columns are from the 2nd column to the second last column (Python index 1 to -1)
feature_cols = df.columns[1:-1]
X_calib = df_calib_cleaned[feature_cols].values
X_test = df_test_cleaned[feature_cols].values
X_all = np.vstack([X_calib, X_test])

# --- 4. Compute calibration statistics (mean and covariance) ---
mean_calib = np.mean(X_calib, axis=0)
cov_calib = np.cov(X_calib, rowvar=False)

# Handle inverse covariance (use pseudo-inverse for robustness)
if cov_calib.ndim == 0:  # Handle single-feature case
    cov_inv = 1.0 / (cov_calib + 1e-10)
else:
    cov_inv = np.linalg.pinv(cov_calib)

# --- 5. Compute sample-level Mahalanobis Distance (MD) ---
def batch_mahalanobis(X, mean, inv_cov):
    delta = X - mean
    # Matrix-form MD computation: sqrt((x-mu)^T * Inv * (x-mu))
    m_dist = np.sqrt(np.einsum('nj,jk,nk->n', delta, inv_cov, delta))
    return m_dist

all_sample_md = batch_mahalanobis(X_all, mean_calib, cov_inv)

# --- 6. Aggregate MD results by day ---
daily_results = []

# Process calibration days
for d in range(calib_days):
    start, end = d * samples_per_day, (d + 1) * samples_per_day
    day_md = np.mean(all_sample_md[start:end])
    daily_results.append({'days': d + 1, 'MD': day_md, 'type': 'calibration'})

# Process test days
offset = calib_days * samples_per_day
for d in range(test_days):
    start, end = offset + d * samples_per_day, offset + (d + 1) * samples_per_day
    day_md = np.mean(all_sample_md[start:end])
    daily_results.append({'days': calib_days + d + 1, 'MD': day_md, 'type': 'test'})

# --- 7. Export results ---
final_df = pd.DataFrame(daily_results)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n✅ MD computation completed. Results saved to: {output_path}")
print(final_df.head())
