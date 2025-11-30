import time
import utils_BiLSTMVersion
import pandas as pd
import os
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pickle files

INPUT_DIR = utils_BiLSTMVersion.RESULTS_DIR / "S1" / "S1.1"
OUTPUT_DIR = utils_BiLSTMVersion.RESULTS_DIR / "S3" / "S3.1" / "S3.1.2" / "BiLSTM_nia"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Reading inputs from: {INPUT_DIR}")
print(f"Saving outputs to: {OUTPUT_DIR}")

train_file_path = INPUT_DIR / "NIAvalueUpdatedVersion_scaled_whole_decedents_dataset.pkl"

if not train_file_path.exists():
    raise FileNotFoundError(f"Input file not found: {train_file_path}\nPlease run S1.1 first.")

scaled_train = pd.read_pickle(train_file_path)

def run_lstm_hyperparameter_selection(feature_columns, target_columns, results_filename):
    start_time = time.time()  # Timing starts

    train_ids = scaled_train.projid.unique()
    train_sequences = utils_BiLSTMVersion.create_sequences(scaled_train, train_ids, feature_columns, target_columns)
    # Move each tensor in the sequences to the device
    train_sequences = [(features.to(device), targets.to(device)) for features, targets in train_sequences] 

    # test_ids = scaled_test.projid.unique()
    # test_sequences = utils_BiLSTMVersion.create_sequences(scaled_test, test_ids, feature_columns, target_columns)
    # # Move each tensor in the sequences to the device
    # test_sequences = [(features.to(device), targets.to(device)) for features, targets in test_sequences]

    # Define hyperparameter grid
    hyperparameter_grid = {
        'hidden_size': [4, 8, 16, 32], 
        'num_layers': [1, 2, 3, 4],
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [16, 32, 64], 
        'dropout_rate': [0, 0.2, 0.4, 0.5]
    }

    # Call the function for hyperparameter selection
    results_df = utils_BiLSTMVersion.select_lstm_hyperparameters(
        train_sequences=train_sequences,
        feature_columns=feature_columns,
        target_columns=target_columns,
        hyperparameter_grid=hyperparameter_grid,
        seed=1217,
        n_splits=5,  # number of cross-validation splits
        num_epochs=500,
        patience=10,
        lr_scheduler_patience=5,
        lr_factor=0.5
    )

    # Save results to a CSV file
    results_df.to_csv(results_filename, index=False)
    end_time = time.time()  # Timing ends
    elapsed_time = end_time - start_time

    print(f"Results saved to {results_filename}")
    print(f"Time elapsed for target(s) {target_columns}: {elapsed_time:.2f} seconds")

# Different target_columns
target_sets = [ 
    ['gpath', 'tangles', 'amyloid', 'niareagansc']
]

# Generate sequences for train and test sets
feature_columns = scaled_train.drop(columns=['projid', 'study', 'fu_year', 'cogdx', 
                                             'amyloid', 'gpath', 'tangles', 'niareagansc']).columns.tolist()

os.makedirs('results', exist_ok=True)
for targets in target_sets:
    file_name = f"Bilstm_nia_{'_'.join(targets)}.csv"
    save_path = OUTPUT_DIR / file_name
    
    run_lstm_hyperparameter_selection(feature_columns, targets, save_path)