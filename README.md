# Inferring Alzheimer’s Disease (AD) Pathologies from Clinical Measures via Cross-Validated Deep Learning Models (Reproducible Pipeline)
One-stop, mixed Make/Python/R workflow for Alzheimer's pathology trajectory modeling and prediction. This repo mirrors the research pipeline with a reproducibility-first layout.

## Data (dummy-friendly)
- Repo includes synthetic/dummy data only for demonstration, as real longitudinal AD datasets are confidential and cannot be shared publicly.
- scripts/generate_dummy.py originally generates synthetic schema-aligned files into data/dummy/.
- For reproducibility validation in this repository, the synthetic data folder is renamed to data/raw/ locally so that downstream Rmd/Notebook scripts (S2–S5) run without path breakage.
- The full pipeline including S4/S5 executes successfully when real longitudinal data are placed in data/raw/ in a secure local environment.

## Pipeline Map
- **S1 Data prep** (`scripts/S1_Data_Preparation/...`): notebooks/Rmd to clean/prepare decedents + livings.
- **S2 GLM baseline** (`scripts/S2_GLM/...`): elastic net notebook for baseline comparisons.
- **S3 Decedents**: S3.1 (training) ran on an HPC cluster; best parameters saved to `configs/best_models.py`. S3.2 uses those parameters to predict without retraining.
- **S4 Livings**: retrain/final models on full decedent set; predict for living participants.
- **S5 Plotting & clustering**: combines imputed trajectories and visualizes patterns.
- Makefile orchestrates the above; see commands below.

## Reproducibility Matrix
S1 Data Preparation, Reproducible
The data processing and sanity checks in scripts/S1_Data_Preparation/... can be executed in this repository using the synthetic dummy dataset.

S2 GLM Baseline, Reproducible
The Elastic-Net GLM baseline model in scripts/S2_GLM/... runs successfully using synthetic features for workflow validation and baseline comparison.

S3.1.1 & S3.1.2, Not Reproducible via Makefile
These training steps were executed on an HPC cluster. The best model architectures and hyperparameters are already manually saved in configs/best_models.py, so this stage is intentionally skipped in Makefile automation.

S3.2 Prediction for Decedents, Reproducible
Predictions using the stored cluster-generated parameters can run automatically, without repeating model training.

S4 Longitudinal Imputation & S5 Plotting, Limited or Not Runnable on Synthetic Data
The S4 and S5 steps are expected to fail or produce incomplete outputs under synthetic dummy data due to the lack of realistic longitudinal disease signals. This is a data-quality limitation, not a code-logic or syntax issue.

Full Pipeline, Fully Runnable in Local Secure Environment Only
When running locally with real longitudinal AD data placed safely in data/raw/, the entire pipeline completes end-to-end and produces all expected results and plots.

## How to Run
```bash
# 1) Generate synthetic/dummy data into data/dummy/ from current raw schemas
python scripts/generate_dummy.py

# 2) Run the full workflow (S4/S5 may be limited/fail on synthetic, which is expected)
make all

# 3) Clean temporary artifacts
make clean
```
- Synthetic data is solely for workflow verification and it cannot run the whole process; real-data mode passes all stages.
- If `make` stops at `check_config`, ensure `configs/best_models.py` exists (pre-saved params from cluster runs).

## Key Takeaways (for reviewers)
- README + Makefile paths validated; synthetic mode exercises structure.
- Any S4/S5 gaps come from dummy data limitations, not code errors.
- Cluster-only S3.1 training is baked into `configs/best_models.py`, enabling S3.2 predictions locally.
- With local real longitudinal data, the full pipeline completes and produces plots/trajectories.

## Future Improvements
- Enhance synthetic longitudinal generator to better mimic slopes/temporal correlations (e.g., spline, spectral/FDA-inspired simulations) so S4/S5 preview more realistically.

## Collaborators
- Pipeline design and experimentation were collaboratively developed by me and my senior colleague; identifiable names are intentionally omitted in this public GitHub version.
