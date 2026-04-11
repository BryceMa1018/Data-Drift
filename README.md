# Drift_detection

This repository provides an implementation for drift detection in sequential data using **Mahalanobis Distance** and **CUSUM control charts**.  
The pipeline includes data preprocessing, statistical modeling, and visualization for detecting distribution shifts over time.

---

## Quick Start

To run the drift detection pipeline, follow the steps below.

---

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
python generating_ood_final.py
```
The execution will automatically generate `MD_result.csv`and place it within the `/data` subfolder.

 Execute the statistical process control analysis and evaluation:
```bash
python spc_final.py
```
### Outputs

- `Evaluation Metrics`: A file named `drift_evaluation_results.csv` will be saved in the `/data` folder.
- `Visualization`: A control chart named `cusum_MD.png` will be generated in the `/drift_charts` directory.


### Validation 

The results shown below are generated based on a simulated dataset for validation purposes. These results demonstrate the algorithm's capability to detect shifts but may differ from the specific clinical metrics reported in the paper due to data privacy (e.g., MIMIC-IV access restrictions).



