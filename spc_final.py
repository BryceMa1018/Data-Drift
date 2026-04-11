import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.svm import OneClassSVM
import sys
import os
import errno

# -------------------------- Basic Settings --------------------------
plt.rcParams["font.family"] = ["Arial", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # Properly display minus sign
plt.rcParams['figure.dpi'] = 100  # Base resolution
plt.rcParams['axes.linewidth'] = 1.2  # Axis line width
plt.rcParams['lines.linewidth'] = 1.5  # Plot line width
plt.rcParams['legend.fontsize'] = 10  # Legend font size
plt.rcParams['axes.labelsize'] = 11  # Axis label font size
plt.rcParams['xtick.labelsize'] = 10  # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 10  # Y-axis tick font size


# -------------------------- Utility Functions --------------------------
def calculate_drift_metrics(anomaly_indices, train_days):
    """
    Compute drift detection evaluation metrics (only ARL retained)

    Parameters:
    - anomaly_indices: array of detected anomaly day indices (0-based)
    - train_days: index where test phase starts

    Returns:
    - {"ARL": arl_value}: number of days from test start to first detected anomaly
    """

    # Initialize ARL as NaN (means no drift detected)
    arl = np.nan

    if len(anomaly_indices) > 0:
        # Only consider alarms triggered during test phase
        detected_in_test = anomaly_indices[anomaly_indices >= train_days]

        if detected_in_test.size > 0:
            # ARL = first alarm index - test start index
            # ARL = 0 means drift detected on the first test day
            arl = int(detected_in_test[0] - train_days)

    return {
        "ARL": arl
    }


def ensure_directory_exists(directory):
    """Ensure directory exists"""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except OSError as e:
        if e.errno != errno.EEXIST:
            print(f"Error: Cannot create directory '{directory}', reason: {str(e)}")
            return False
    return True


# -------------------------- Data Loading --------------------------
def load_spc_data(file_path=None):
    """
    Load SPC data (from MD_result.csv)

    Modifications:
    1. Default path is set to data/MD_result.csv
    2. Check whether columns "MD" and "type" exist
    """
    if file_path is None:
        file_path = os.path.join(".", "data", "MD_result.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"SPC data file {file_path} does not exist! Please ensure preprocessing has generated MD_result.csv in the data folder."
        )

    df = pd.read_csv(file_path)

    required_cols = ["MD", "type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"SPC data is missing required columns: {', '.join(missing_cols)}")

    return df


# -------------------------- CUSUM Method --------------------------
def calculate_upper_cusum_limit(normal_data_std, k=0.5, confidence=0.9973, u_max=15, n_points=1500, tol=1e-3,
                                max_iter=500):

    # -------------------------- 1. Fit PDF/CDF --------------------------
    kde = stats.gaussian_kde(normal_data_std)

    pdf_x_range = np.linspace(-u_max - k, u_max + k, n_points)
    pdf_y = kde.evaluate(pdf_x_range)

    def pdf_fun(x):
        return np.interp(x, pdf_x_range, pdf_y, left=pdf_y[0], right=pdf_y[-1])

    def empirical_cdf(z):
        return np.mean(normal_data_std <= z)

    F = np.vectorize(empirical_cdf)

    # -------------------------- 2. Generate grid --------------------------
    u = np.linspace(0, u_max, n_points)
    du = u[1] - u[0]

    target_tail_prob = 1 - confidence

    # -------------------------- 3. Iterative solving --------------------------
    p_old = 1 - F(u + k)
    kernel = pdf_fun(u + k)

    for iteration in range(max_iter):

        conv = np.convolve(p_old, kernel, mode='full')[:n_points] * du
        p_new = 1 - F(u + k) + conv

        if np.nanmax(np.abs(p_new - p_old)) < tol:
            break

        p_old = p_new

    else:
        print("Warning: CUSUM upper limit iteration not fully converged.")
        return 3.0

    # -------------------------- 4. Interpolation --------------------------
    sort_idx = np.argsort(p_new)

    sorted_p = p_new[sort_idx]
    sorted_u = u[sort_idx]

    upper_limit = np.interp(
        target_tail_prob,
        sorted_p,
        sorted_u,
        left=sorted_u[0],
        right=sorted_u[-1]
    )

    return max(upper_limit, 0)


def calculate_lower_cusum_limit(normal_data_std, k=0.5, confidence=0.9973, u_max=15, n_points=1500, tol=1e-3,
                                max_iter=500):
    """
    Fully aligned with R version calc_ucl_empirical:
    Fixes non-convergence issue of CUSUM-, formula matches exactly
    """

    # -------------------------- 1. Fit PDF/CDF --------------------------
    kde = stats.gaussian_kde(normal_data_std)
    pdf_x_range = np.linspace(-u_max - k, u_max + k, n_points)
    pdf_y = kde.evaluate(pdf_x_range)

    def pdf_fun(x):
        return np.interp(x, pdf_x_range, pdf_y, left=pdf_y[0], right=pdf_y[-1])

    def empirical_cdf(z):
        return np.mean(normal_data_std <= z)

    F = np.vectorize(empirical_cdf)

    # -------------------------- 2. Generate grid --------------------------
    u = np.linspace(0, u_max, n_points)
    du = u[1] - u[0]
    target_tail_prob = 1 - confidence

    # -------------------------- 3. Iteration --------------------------
    p_old = F(-k - u)
    kernel = pdf_fun(-u - k)

    p_old = np.clip(p_old, 1e-8, 1 - 1e-8)

    for iteration in range(max_iter):
        conv = np.convolve(p_old, kernel, mode='full')[:n_points] * du
        p_new = F(-k - u) + conv
        p_new = np.clip(p_new, 1e-8, 1 - 1e-8)

        if np.all(np.isfinite(p_new)) and np.nanmax(np.abs(p_new - p_old)) < tol:
            break

        p_old = p_new

    else:
        print("Warning: CUSUM lower limit iteration not fully converged (try max_iter=1000)")
        return 3.0

    # -------------------------- 4. Interpolation --------------------------
    sort_idx = np.argsort(p_new)
    sorted_p = p_new[sort_idx]
    sorted_u = u[sort_idx]

    lower_limit = np.interp(
        target_tail_prob,
        sorted_p,
        sorted_u,
        left=sorted_u[0],
        right=sorted_u[-1]
    )

    return max(lower_limit, 0)


def analyze_cusum(df, metric="MD", k=0.5, confidence=0.9973):
    """CUSUM analysis (MD only), CUSUM- plotted on negative axis"""

    train_days = len(df[df['type'] == 'calibration'])
    normal_data = df.iloc[:train_days][metric].values
    all_data = df[metric].values

    # Normalize data
    mu = np.mean(normal_data)
    sigma = np.std(normal_data)
    sigma = sigma if sigma != 0 else 1e-8
    normal_data_std = (normal_data - mu) / sigma
    all_data_std = (all_data - mu) / sigma

    # Control limits
    upper_control_limit = calculate_upper_cusum_limit(normal_data_std, k, confidence)
    lower_control_limit = calculate_lower_cusum_limit(normal_data_std, k, confidence)

    # Compute CUSUM
    n = len(all_data_std)
    cusum_upper = np.zeros(n)
    cusum_lower = np.zeros(n)

    for i in range(n):
        if i == 0:
            cusum_upper[i] = max(0, all_data_std[i] - k)
            cusum_lower[i] = max(0, -all_data_std[i] - k)
        else:
            cusum_upper[i] = max(0, cusum_upper[i - 1] + all_data_std[i] - k)
            cusum_lower[i] = max(0, cusum_lower[i - 1] - all_data_std[i] - k)

    cusum_lower_neg = -cusum_lower
    lower_control_limit_neg = -lower_control_limit

    upper_anomalies = np.where(cusum_upper > upper_control_limit)[0]
    lower_anomalies = np.where(cusum_lower > lower_control_limit)[0]
    anomalies = np.union1d(upper_anomalies, lower_anomalies)
    metrics = calculate_drift_metrics(anomalies, train_days)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(range(1, n + 1), cusum_upper, 'forestgreen', marker='^', markersize=5,
            label='CUSUM+ (Upper)', markerfacecolor='lightgreen', markeredgecolor='forestgreen')

    ax.plot(range(1, n + 1), cusum_lower_neg, 'crimson', marker='v', markersize=5,
            label='CUSUM- (Lower)', markerfacecolor='lightcoral', markeredgecolor='crimson')

    ax.axhline(upper_control_limit, color='forestgreen', linestyle='--', linewidth=1.2,
               label=f'UCL ({confidence * 100:.2f}%): {upper_control_limit:.4f}')
    ax.axhline(lower_control_limit_neg, color='crimson', linestyle='--', linewidth=1.2,
               label=f'LCL ({confidence * 100:.2f}%): {lower_control_limit_neg:.4f}')

    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    ax.axvline(x=train_days + 0.5, color='orange', linestyle=':', linewidth=1.5,
               label='Beginning of Test Set')

    ax.scatter(upper_anomalies + 1, cusum_upper[upper_anomalies], color='orange', s=80,
               facecolors='none', edgecolors='orange', linewidth=2,
               label='Detected Upper Anomalies')

    ax.scatter(lower_anomalies + 1, cusum_lower_neg[lower_anomalies], color='purple', s=80,
               facecolors='none', edgecolors='purple', linewidth=2,
               label='Detected Lower Anomalies')

    ax.set_title(f"CUSUM Control Chart - Mahalanobis Distance\n"
                 f"ARL: {metrics['ARL'] if not np.isnan(metrics['ARL']) else 'N/A'}")

    ax.set_xlabel("Days")
    ax.set_ylabel("CUSUM Statistic")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return fig, anomalies, metrics


# -------------------------- Main Function --------------------------
def main():
    chart_dir = "drift_charts"
    ensure_directory_exists(chart_dir)
    results = []

    try:
        df_spc = load_spc_data()
        print("✅ SPC data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load SPC data: {str(e)}")
        return

    try:
        fig, anomalies, metrics = analyze_cusum(df_spc)
        fig.savefig(f"{chart_dir}/cusum_MD.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        results.append({"Method": "CUSUM", "Metric": "Mahalanobis Distance (MD)", **metrics})
        print("✅ CUSUM analysis completed, chart saved")
    except Exception as e:
        print(f"❌ CUSUM analysis failed: {str(e)}")

    if results:
        out_csv = "data\\drift_evaluation_results.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ All results saved to: {os.path.abspath(out_csv)}")

        print("\n📊 Evaluation Results Preview:")
        print(pd.DataFrame(results).to_string(index=False))
    else:
        print("\n❌ No results generated, please check data and methods")


if __name__ == "__main__":
    main()