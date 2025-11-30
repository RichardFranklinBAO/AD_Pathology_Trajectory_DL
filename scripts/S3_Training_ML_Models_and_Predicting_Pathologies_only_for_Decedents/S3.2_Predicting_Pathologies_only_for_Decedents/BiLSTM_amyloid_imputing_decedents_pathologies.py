import time
import pandas as pd
import torch
import sys
from pathlib import Path

# 1. Set up Project Root & Imports
# Get current file path
current_file = Path(__file__).resolve()

# --- Add 'models' parent directory to sys.path (S3.2 folder) ---
models_parent_dir = current_file.parent
sys.path.append(str(models_parent_dir))

# --- Add Project Root to sys.path ---
# Navigate up 5 levels: S3.2 -> S3 -> scripts -> Root
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

# Verify paths (Debugging)
print(f"Added models parent dir: {models_parent_dir}")
print(f"Added project root: {project_root}")

# Import utilities
from models import utils_training

# Import Model Architectures from your new 'models' folder
from models.bilstm import BiLSTM
from models.lstm import LSTM
from models.lstm_relu import LSTMReLU

# Import Config
from configs.best_models import MODEL_PARAMS

# Import Configuration
from configs.best_models import MODEL_PARAMS

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. Define Reproducible Paths
# ==========================================

INPUT_DIR = utils_training.RESULTS_DIR / "S1" / "S1.1"
# Output directory for Amyloid predictions
OUTPUT_DIR = utils_training.RESULTS_DIR / "S3" / "S3.2" / "BiLSTM_amyloid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Reading inputs from: {INPUT_DIR}")
print(f"Saving outputs to: {OUTPUT_DIR}")

# ==========================================
# 2. Load Data
# ==========================================

train_file_path = INPUT_DIR / "NIAvalueUpdatedVersion_scaled_whole_decedents_dataset.pkl"
if not train_file_path.exists():
    raise FileNotFoundError(f"Input file not found: {train_file_path}")

scaled_train = pd.read_pickle(train_file_path)
print(f"Successfully loaded data: {train_file_path.name}")

# ==========================================
# 3. Load Configuration & Model Factory
# ==========================================

# Identify pathology
PATHOLOGY_NAME = "amyloid"

if PATHOLOGY_NAME not in MODEL_PARAMS:
    raise ValueError(f"Pathology {PATHOLOGY_NAME} not found in config!")

CONFIG = MODEL_PARAMS[PATHOLOGY_NAME]
ARCH = CONFIG["architecture"]
HYPERPARAMS = CONFIG["hyperparameters"]
TARGETS = CONFIG["target_columns"]

print(f"Loaded Config for {PATHOLOGY_NAME}: Arch={ARCH}, Params={HYPERPARAMS}")

def get_model_instance(arch_name, input_dim, output_dim, hparams):
    hidden = hparams['hidden_size']
    layers = hparams['num_layers']
    drop = hparams['dropout_rate']
    
    if arch_name == "BiLSTM":
        return BiLSTM(input_dim, hidden, output_dim, layers, drop, bidirectional=True)
    elif arch_name == "LSTM":
        return LSTM(input_dim, hidden, output_dim, layers, drop, bidirectional=False)
    elif arch_name == "LSTMReLU":
        return LSTMReLU(input_dim, hidden, output_dim, layers, drop)
    else:
        raise ValueError(f"Unknown Architecture: {arch_name}")

# ==========================================
# 4. Training Execution
# ==========================================

def run_imputation():
    start_time = time.time()

    train_ids = scaled_train.projid.unique()
    
    # 1. Create Sequences
    # Feature columns: All columns minus metadata and ALL pathology targets (to avoid leakage)
    all_targets = ['amyloid', 'gpath', 'tangles', 'niareagansc']
    drop_cols = ['projid', 'study', 'fu_year', 'cogdx'] + all_targets
    feature_columns = scaled_train.drop(columns=drop_cols).columns.tolist()
    
    train_sequences = utils_training.create_sequences(
        scaled_train, train_ids, feature_columns, TARGETS
    )
    train_sequences = [(f.to(device), t.to(device)) for (f, t) in train_sequences]

    # 2. Initialize Model
    input_dim = len(feature_columns)
    output_dim = len(TARGETS)
    
    model_instance = get_model_instance(ARCH, input_dim, output_dim, HYPERPARAMS)
    model_instance.to(device)

    # 3. Run Training
    # Prepare single-item grid for the training function
    grid = {
        'hidden_size': [HYPERPARAMS['hidden_size']],
        'num_layers': [HYPERPARAMS['num_layers']],
        'learning_rate': [HYPERPARAMS['learning_rate']],
        'batch_size': [HYPERPARAMS['batch_size']],
        'dropout_rate': [HYPERPARAMS['dropout_rate']]
    }
    
    print(f"Starting training for targets: {TARGETS} using {ARCH}...")

    # Pass 'bidirectional' flag explicitly for logic inside training utils if needed
    is_bidirectional = (ARCH == "BiLSTM")

    results_df, oof_df = utils_training.select_lstm_hyperparameters(
        model=model_instance,  # Pass the initialized model instance
        train_sequences=train_sequences,
        feature_columns=feature_columns,
        target_columns=TARGETS,
        hyperparameter_grid=grid,
        seed=1217,
        n_splits=5,
        num_epochs=500,
        patience=10,
        lr_scheduler_patience=5,
        lr_factor=0.5,
        train_ids=train_ids,             
        df_all=scaled_train,             
        device=device,
        bidirectional=is_bidirectional
    )

    # 4. Save Results
    prefix = f"BiLSTM_amyloid_{'_'.join(TARGETS)}"
    results_df.to_csv(OUTPUT_DIR / f"{prefix}.csv", index=False)
    
    target_str = '_'.join(TARGETS)
    oof_df.sort_values(["projid", "fu_year"]).to_csv(
        OUTPUT_DIR / f"BiLSTM_amyloid_clean_imputed_{target_str}_cv5_allsteps.csv", index=False
    )

    print(f"Done. Elapsed: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    run_imputation()