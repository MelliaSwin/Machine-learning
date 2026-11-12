# Gambling Impulsivity (BiLSTM Autoencoder)
**By Melissa Swinehart**

---

## Overview
For my project, I implemented a **BiLSTM Autoencoder** to detect impulsive/anomalous
gambling behavior from transaction sequences.
The model learns normal behavior and assigns anomaly scores based on reconstruction
error. The higher the reconstruction error, the more impulsive activity.

---

## Project Structure
casino-bilstm/
- README.MD
- Subfolder: src/ <br>
  - outputs/ 
    - Contains visual graphs, charts, and comparisons between customers
      - Normal_Customers/ : 5 Normal Predictable Customers and their visuals
      - Risk_Customers/   : Top 5 'at risk' customers and their visuals
      - __.png images here as well
  - config.py
    - Change hyperparameters and global settings
  - dataset.py
    - Preprocessing + Sequence Creation
  - model.py
    - BiLSTM Autoencoder Architecture
  - train.py
    - Training loop + Evaluation
  - main.py
    - Runs full pipeline and produces visuals
  - baselines.py
    - Optional models for comparison (mainly used for report)

## How to Run
From inside the 'src/' directory:
```bash
python main.py
```

## Model Summary
- BiLSTM Encoder (Forward + Backward)
- MSE loss for reconstruction

## Baseline Models include:
- Isolation Forest
- K-Means Clustering

# Visualizations
From inside the 'outputs/' directory:
- Histogram of anomaly scores (anomaly_histogram.png)
- High risk customers vs normal customers comparison (customer_comparison.png)
- Timeline plots for selected customers (subfolders)
- Training Loss Curve (training_loss.png)

---
>**Note:**
> The 'outputs/' folder already contains visualizations, graphs, and timeline plots
> for 5 high-risk customers and 5 normal customers
> 
> When you run the model again using 'python main.py', the script will:
> - Overwrite the existing summary plots (e.g., histogram, comparison chart), and
> - generate **5 new high risk customers** and **5 new normal customers**,
> - which will be added to the existing subfolders inside 'outputs/'.
> 
> This behavior is expected and **DOES NOT** affect the functionality of the project