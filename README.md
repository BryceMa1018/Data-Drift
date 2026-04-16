# Drift_detection

## 📖 Background

In the dynamic environment of healthcare, clinical predictive models often suffer from performance silent failure due to "data drift". This occurs when the distribution of inputs and outputs changes over time, rendering static models unreliable.

This repository implements a robust **Statistical Process Control (SPC)** framework. By mapping high-dimensional feature into **Mahalanobis Distance (MD)** space and monitoring them via **Cumulative Sum (CUSUM)** control charts, our approach provides a highly sensitive, real-time mechanism to detect subtle performance decays before they compromise patient safety.

---

## 📊 Quick Start

To run the drift detection pipeline, follow the steps below.

---

### 🛠️ Environment Installation

This project is developed and tested using **Python 3.13+**. The following core libraries are required to run the detection pipeline:

| Library | Full Name | Version Used |
| :--- | :--- | :--- |
| **NumPy** | Numerical Python | 2.1.3 |
| **Pandas** | Python Data Analysis Library | 2.2.3 |
| **Matplotlib** | Mathematical Plotting Library | 3.10.0 |
| **SciPy** | Scientific Python | 1.15.3 |

To ensure full compatibility, we recommend installing the exact versions specified in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Data Preparation

1. Prepare your raw dataset and place it under the `./data/` directory.

2. The dataset should include at least the following columns:
   - `sample_id`: must be continuous positive integers starting from 1 to N (where N is the total number of samples).
   - `dataset_type`: must be either `calibration` or `test`.
     
       -`calibration`: Represents baseline data under normal conditions. These samples must appear first in the file.
     
       -`test`: Data to be monitored for potential drift. These samples must follow the calibration data and cannot be interleaved. 
   - numerical feature columns

The first column must be sample_id, and the last column must be dataset_type.
   
3. Configure the dataset settings in:
   ```bash
   configuration.json
   ```
### Run the code
 Execute the following command:
```bash
python generating_MD.py
```
The execution will automatically generate `MD_result.csv`and place it within the `/data` subfolder.

 Execute the statistical process control analysis and evaluation:
```bash
python spc.py
```
### Outputs

- `Evaluation Metrics`: A file named `drift_evaluation_results.csv` will be saved in the `/data` folder.
- `Visualization`: A control chart named `cusum_MD.png` will be generated in the `/drift_charts` directory.





---

## 🧪 Experimental Validation & Results

To demonstrate the reliability of the detection pipeline, we conducted a comprehensive simulation using the provided scripts.

### 1. Validation Process
The validation follows a structured two-stage workflow using a synthetic dataset of **1,000 samples** with a **bivariate standard normal distribution**:

* **Temporal Scale**: For realistic interpretation, the data is partitioned into daily intervals, with **20 samples representing 1 day** (totaling 50 days of monitoring).
* **Baseline Calibration**: The first 500 samples (Days 1–25, labeled as `calibration`) are used to fit the empirical distribution of Mahalanobis Distances. This stage establishes the "normal" operational profile under stable conditions.
* **Drift Monitoring**: The subsequent 500 samples (Days 26–50, labeled as `test`) are used for real-time monitoring. To simulate real-world data drift, a **mean shift** was introduced starting from the "test" phase, while maintaining the original covariance structure.



### 2. Results Display


The results shown below are generated based on a simulated dataset for validation purposes. These results demonstrate the algorithm's capability to detect shifts but may differ from the specific clinical metrics reported in the paper due to data privacy (e.g., MIMIC-IV access restrictions).

The analysis yields two primary outputs that can be found in this repository:

#### **A. CUSUM Control Chart**
The visualization below (stored in `/drift_charts`) illustrates the core detection logic. The **blue and red lines** represent the cumulative sum of positive and negative deviations. When either line crosses the **dashed control limits**, a drift alarm is triggered.

![CUSUM Control Chart](drift_charts/cusum_MD.png)

#### **B. Quantitative Evaluation**
The performance of the drift detection pipeline is evaluated using the Average Run Length (ARL), which measures the delay between the drift onset and the first alarm. The result is exported to `data/drift_evaluation_results.csv`. 


| Method | Metric | ARL | 
| :--- | :--- | :--- | 
| CUSUM | Mahalanobis Distance (MD) | 0  | 


* **Detection Delay**: Our results indicate that the drift was successfully captured within a very short interval after the onset, proving the high sensitivity of the Mahalanobis-CUSUM approach.

---


## 📚 Citation & References
### Cite this work

If you use this code or our  CUSUM-SPC framework in your research, please cite our paper:

Ma, J., et al. (2026). Algorithm's Performance Detection. Journal of Biomedical Informatics (In Press/Submitted).











