import os
import tempfile
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torchmetrics import R2Score

# device = "cuda" if torch.cuda.is_available() else "cpu"

# utils.py
import sys
from pathlib import Path

def get_project_root():
    """
    Automatically locate the project root directory (AD_Github_Version).
    Logic: Start from the current file and search upwards until 'Makefile' is found.
    """
    # Get the absolute path of the current utils.py file
    current_path = Path(__file__).resolve()
    
    # Loop upwards through parent directories
    for parent in [current_path] + list(current_path.parents):
        if (parent / "Makefile").exists():
            return parent
            
    # Fallback: if Makefile is missing, look for the 'data' folder
    if (current_path / "data").exists():
        return current_path
        
    raise FileNotFoundError("Error: Project root not found! Please ensure a Makefile exists in the root directory.")

# 1. Get the Project Root object
ROOT = get_project_root()

# 2. Define standard directory shortcuts
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SCRIPTS_DIR = ROOT / "scripts"

# 3. Print paths for verification (when running this script directly)
if __name__ == "__main__":
    print(f"Project Root: {ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    

# Utility functions
# --- NEW: 为全时间步推理准备的序列与 collate ---

def create_sequences_with_ids(df, ids, feature_cols, target_cols):
    """
    return (projid, feature_seq, target_seq, fu_year_seq)
    - feature_seq: (T, in_dim) float32
    - target_seq:  (T, out_dim) float32 
    - fu_year_seq: (T,) numpy
    """
    seqs = []
    for pid in ids:
        sub = df[df['projid'] == pid].sort_values('fu_year')
        X = sub[feature_cols].values.astype('float32')
        Y = sub[target_cols].values.astype('float32')
        FY = sub['fu_year'].values  # keep as numpy
        seqs.append((pid, X, Y, FY))
    return seqs


class VariableLengthTensorDatasetWithIds(Dataset):
    """A Dataset that stores (projid, X_seq, Y_seq, fu_year_seq); used only for inference DataLoader."""
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx]


def custom_collate_with_ids(batch):
    """
    batch: list of (projid, X_seq_np, Y_seq_np, fu_year_np)
    return：
      - projids: list[str/int]
      - packed_X: PackedSequence
      - lens: list[int]  
      - fu_years: list[np.ndarray] 
    """
    projids   = [b[0] for b in batch]
    X_list    = [torch.tensor(b[1]) for b in batch]
    # Y_list    = [torch.tensor(b[2]) for b in batch]  # 不一定用到
    fu_years  = [b[3] for b in batch]

    lengths = [x.shape[0] for x in X_list]
    sort_idx = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)

    projids  = [projids[i]  for i in sort_idx]
    fu_years = [fu_years[i] for i in sort_idx]
    X_list   = [X_list[i]   for i in sort_idx]
    # Y_list = [Y_list[i]   for i in sort_idx]  # 如需要可保留

    packed_X = torch.nn.utils.rnn.pack_sequence(X_list, enforce_sorted=True)
    return projids, packed_X, lengths, fu_years

# --- NEW: All-time-steps 头。与训练结构同名层，forward 输出 (B, T, out_dim) ---

class LSTMAllStepsHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True, bidirectional=bidirectional
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim * self.num_dirs)
        self.fc = nn.Linear(hidden_dim * self.num_dirs, output_dim)

    def forward(self, packed_x):
        packed_out, _ = self.lstm(packed_x)
        padded_out, lens = pad_packed_sequence(packed_out, batch_first=True)      # (B, T, H*)
        normed = self.batch_norm(padded_out.transpose(1, 2)).transpose(1, 2)      # (B, T, H*)
        preds = self.fc(normed)                                                   # (B, T, out_dim)
        return preds, lens


def infer_all_timesteps_df(trained_model, df_all, test_ids, feature_cols, target_cols,
                           hidden_size, num_layers, dropout_rate, bidirectional=True, device="cpu"):
    # 1) Construct sequences with IDs
    seqs = create_sequences_with_ids(df_all, test_ids, feature_cols, target_cols)
    loader = DataLoader(
        VariableLengthTensorDatasetWithIds(seqs),
        batch_size=len(seqs), shuffle=False, collate_fn=custom_collate_with_ids
    )

    # 2) Use the all-timesteps head and load the same weights
    all_steps = LSTMAllStepsHead(
        input_dim=len(feature_cols),
        hidden_dim=hidden_size,
        output_dim=len(target_cols),
        num_layers=num_layers,
        dropout=dropout_rate,
        bidirectional=bidirectional
    ).to(device)
    all_steps.load_state_dict(trained_model.state_dict())
    all_steps.eval()

    # 3) Run inference and aggregate results
    out_rows = []
    with torch.no_grad():
        for projids, packed_X, lens, fu_years in loader:
            preds, lens_tensor = all_steps(packed_X.to(device))
            preds = preds.cpu().numpy()  # (B, T, out_dim)
            lens  = lens_tensor.cpu().numpy().tolist() 

            for i, pid in enumerate(projids):
                T = int(lens[i])
                years = fu_years[i][:T]
                p = preds[i, :T, :]  # (T, out_dim)

                for t in range(T):
                    row = {"projid": pid, "fu_year": years[t]}
                    for j, tgt in enumerate(target_cols):
                        row[f"pred_{tgt}"] = float(p[t, j])
                    out_rows.append(row)

    df_pred = pd.DataFrame(out_rows).sort_values(["projid", "fu_year"]).reset_index(drop=True)
    return df_pred


def load_and_merge_data(features_path, targets_path, join_columns):
    """
    Load and merge datasets based on the specified join columns.

    Parameters:
        features_path (str): Path to the features file.
        targets_path (str): Path to the targets file.
        join_columns (list): Columns to join on.

    Returns:
        pd.DataFrame: Merged dataset.
    """
    try:
        features = pd.read_csv(features_path, sep="\t")
        targets = pd.read_csv(targets_path, sep="\t")
        return pd.merge(features, targets, on=join_columns)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise


def transform_data(data, columns, transformer_type="StandardScaler", transformers=None, fit=True, **kwargs):
    """
    Transform specified columns of a DataFrame with the chosen scikit-learn transformer.

    Parameters:
        data (pd.DataFrame): Input data.
        columns (list): List of columns to transform.
        transformer_type (str): The type of transformer to use (e.g., "StandardScaler", "MinMaxScaler").
        transformers (dict): Pre-fitted transformers for transforming data (if fit=False).
        fit (bool): Whether to fit new transformers or use existing ones.
        **kwargs: Additional keyword arguments to pass to the transformer.

    Returns:
        Tuple[pd.DataFrame, dict]: Transformed data and fitted transformers (if fit=True).
    """
    # Dictionary to map transformer_type string to actual scikit-learn classes
    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        RobustScaler,
        MaxAbsScaler,
        Normalizer,
    )
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold

    transformer_map = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
        "MaxAbsScaler": MaxAbsScaler,
        "Normalizer": Normalizer,
        "PCA": PCA,
        "VarianceThreshold": VarianceThreshold,
    }

    if transformer_type not in transformer_map:
        raise ValueError(
            f"Unsupported transformer_type '{transformer_type}'. "
            f"Choose from {list(transformer_map.keys())}."
        )

    # Initialize a copy of the input DataFrame
    transformed_data = data.copy()

    if fit:  # Fit and transform
        transformers = {}
        for col in columns:
            transformer = transformer_map[transformer_type](**kwargs)  # Create a new transformer instance
            transformed_data[col] = transformer.fit_transform(data[[col]])
            transformers[col] = transformer
        return transformed_data, transformers
    else:  # Transform only
        if transformers is None:
            raise ValueError("Transformers must be provided if fit=False.")
        for col in columns:
            if col not in transformers:
                raise ValueError(f"Transformer for column '{col}' is not provided.")
            transformed_data[col] = transformers[col].transform(data[[col]])
        return transformed_data


def get_last_visit(data, sort_columns, group_column):
    """
    Retrieves the last entry for each group based on the specified sort order.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data.
        sort_columns (list): List of column names to sort the data by.
        group_column (str): The column name to group the data by.

    Returns:
        pd.DataFrame: A DataFrame containing the last entry for each group.
    """
    # Sort the DataFrame by the specified columns to ensure the correct order
    sorted_data = data.sort_values(by=sort_columns)

    # Group the sorted data by the specified column and take the last entry for each group
    last_visits = sorted_data.groupby(group_column).last()

    # Reset the index to return a standard DataFrame (rather than a grouped DataFrame)
    result = last_visits.reset_index()

    return result


class ElasticNetRidgeSwitcher(BaseEstimator, RegressorMixin):
    """
    Custom estimator to switch between Ridge and ElasticNet regression.

    Parameters:
        alpha (float): Regularization strength.
        l1_ratio (float): ElasticNet mixing parameter. l1_ratio=0 corresponds to Ridge regression.
        max_iter (int): Maximum number of iterations.

    Attributes:
        model_ (object): The trained Ridge or ElasticNet model.
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=10000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray): Target vector.

        Returns:
            self
        """
        if self.l1_ratio == 0:
            self.model_ = Ridge(alpha=self.alpha)
        else:
            self.model_ = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the trained model.

        Parameters:
            X (pd.DataFrame or np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predictions.
        """
        return self.model_.predict(X)


def create_sequences(dataframe, ids, features, targets):
    """
    Create sequences and targets for PyTorch models.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame containing all data.
        ids (array-like): Unique IDs for splitting data.
        features (list): Columns to include in the input sequence.
        targets (list): Columns to include as target values.

    Returns:
        list of tuples: [(input_tensor, target_tensor), ...]
    """
    sequences = []

    for id in ids:
        # Filter rows for the current ID
        tmp_df = dataframe[dataframe.projid == id]

        # Extract and convert target columns to a tensor
        target_values = tmp_df[targets].iloc[0].values
        target_tensor = torch.tensor(target_values, dtype=torch.float32)

        # Extract and convert feature columns to a tensor
        feature_values = tmp_df[features].to_numpy()
        feature_tensor = torch.tensor(feature_values, dtype=torch.float32)

        sequences.append((feature_tensor, target_tensor))

    return sequences


# Custom Dataset for variable-length sequences
class VariableLengthTensorDataset(Dataset):
    """
    A custom dataset for handling variable-length sequences.
    """
    def __init__(self, tensor_list, ids = None):
        """
        Args:
            tensor_list (list): A list of (sequence, target) tuples.
            ids: list of projid，与 tensor_list 一一对应；不传则用 0..N-1
        """
        self.tensor_list = tensor_list
        if ids is None:
            self.ids = list(range(len(tensor_list)))
        else:
            assert len(ids) == len(tensor_list), "ids 与数据长度不一致"
            self.ids = list(ids)

    def __len__(self):
        return len(self.tensor_list)

    # def __getitem__(self, idx):
    #     return self.tensor_list[idx]
    
    def __getitem__(self, idx):
        seq, tgt = self.tensor_list[idx]
        return seq, tgt, self.ids[idx]


# Custom collate function for handling variable-length sequences
def custom_collate(batch):
    """
    Custom collate function to pad sequences and prepare data for LSTM.

    Args:
        batch (list): A batch of (sequence, target, projid) tuples.

    Returns:
        Tuple[PackedSequence, targets, pids_sorted]: Packed padded sequences and corresponding targets.
    """
    sequences, targets, pids = zip(*batch)
    
    # Sort sequences by length in descending order
    # sequences, targets = zip(*sorted(zip(sequences, targets), key=lambda x: len(x[0]), reverse=True))

    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True)
    sequences = [sequences[i] for i in order]
    targets   = [targets[i]   for i in order]
    pids      = [pids[i]      for i in order]
    
    lengths = [len(seq) for seq in sequences]  # Sequence lengths
    # Pad sequences and create packed sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)
    packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=True)
    # Stack targets
    targets = torch.stack(targets)
    return packed_sequences, targets, pids



def compute_r2(predictions, ground_truth):

    return np.corrcoef(predictions, ground_truth)[1, 0] ** 2


def pearson_correlation(y_pred, y_true):
    """Compute Pearson correlation between predictions and true values."""
    mean_pred = torch.mean(y_pred, dim=0, keepdim=True)
    mean_true = torch.mean(y_true, dim=0, keepdim=True)

    num = torch.sum((y_pred - mean_pred) * (y_true - mean_true), dim=0)
    denom = torch.sqrt(
        torch.sum((y_pred - mean_pred) ** 2, dim=0) *
        torch.sum((y_true - mean_true) ** 2, dim=0)
    )

    pearson_corr = num / (denom + 1e-8)  # Avoid division by zero
    return pearson_corr


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, y_pred, y_true):
        pearson_corr = pearson_correlation(y_pred, y_true)
        return 1 - pearson_corr.sum()  # Minimize (1 - Pearson correlation)

     
     
def train_and_evaluate_model(
        model, train_data, test_data, input_dim, output_dim, hidden_size, num_layers, learning_rate, batch_size,
        num_epochs=100, patience=10, lr_scheduler_patience=5, lr_factor=0.5, test_size=0.2, random_state=42,
        seed=1217, model_save_path="best_model.pth", temporary=False,
        weight_decay=1e-5, dropout_rate=0.5, *, target_names, train_ids=None, test_ids=None):
    """
    Function for training and evaluating a model with optional temporary saving of the best model.

    Args:
        train_data (list): Training dataset [(sequence, target), ...].
        test_data (list): Test dataset [(sequence, target), ...].
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        hidden_size (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        learning_rate (float): Learning rate for optimization.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        patience (int): Early stopping patience.
        lr_scheduler_patience (int): Patience for learning rate scheduler.
        lr_factor (float): Factor for reducing the learning rate.
        test_size (float): Proportion of the training data used as validation data.
        random_state (int): Random state for reproducibility.
        seed (int): Random seed for reproducibility.
        model_save_path (str): Path to save the best model.
        temporary (bool): If True, saves the model to a temporary file that is deleted afterward.

    Returns:
        dict: R-squared scores for each outcome on the test dataset.
        list: Training loss history.
        list: Validation loss history.
        list: Learning rate history.
    """
    from sklearn.model_selection import train_test_split

    # Split train_data into train and validation sets
    train_idx, val_idx = train_test_split(range(len(train_data)), test_size=test_size, random_state=random_state)
    train_split = [train_data[i] for i in train_idx]
    val_split = [train_data[i] for i in val_idx]
    
    train_ids_split = [train_ids[i] for i in train_idx] if train_ids is not None else None
    val_ids_split   = [train_ids[i] for i in val_idx]   if train_ids is not None else None


    torch.manual_seed(seed)
    # Create dataloaders
    train_loader = DataLoader(VariableLengthTensorDataset(train_split, train_ids_split), batch_size=batch_size, collate_fn=custom_collate, shuffle=True)
    val_loader = DataLoader(VariableLengthTensorDataset(val_split, val_ids_split), batch_size=len(val_idx), collate_fn=custom_collate, shuffle=False)
    test_loader = DataLoader(VariableLengthTensorDataset(test_data, test_ids), batch_size=len(test_data), collate_fn=custom_collate, shuffle=False)

    # Set the random seed
    # torch.manual_seed(seed)

    # Initialize model, loss, optimizer, and scheduler
    # model = LSTMModel(input_dim, hidden_size, output_dim, num_layers, dropout_rate) #.to(device)

    criterion = PearsonCorrelationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=lr_scheduler_patience, factor=lr_factor)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Lists to record losses and learning rate
    training_loss_history = []
    validation_loss_history = []
    learning_rate_history = []

    # Determine save path (temporary or fixed)
    if temporary:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_model_path = temp_file.name
        temp_file.close()  # Close the file so PyTorch can write to it
    else:
        temp_model_path = model_save_path

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if len(batch) == 3:
                batch_x, batch_y, _ = batch 
            else:
                batch_x, batch_y = batch
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        training_loss_history.append(train_loss)

        # Evaluate on the validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    batch_x, batch_y, _ = batch
                else:
                    batch_x, batch_y = batch
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        validation_loss_history.append(val_loss)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Log the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rate_history.append(current_lr)

        # Save the model if validation loss is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), temp_model_path)  # Save the model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load the best model for evaluation
    model.load_state_dict(torch.load(temp_model_path))

    # Evaluate on the test set

    model.eval()
    r2_scores = []
    pred_dfs = []
    with torch.no_grad():
        for batch_x, batch_y, pids in test_loader:
            outputs = model(batch_x)
            yhat = outputs.cpu().numpy()
            ytrue = batch_y.cpu().numpy()

            # R2：following column
            for i in range(output_dim):
                r2_scores.append(compute_r2(yhat[:, i], ytrue[:, i]))

            df_batch = pd.DataFrame(yhat, columns=[f"pred_{t}" for t in target_names])
            df_batch.insert(0, "projid", list(pids))
            pred_dfs.append(df_batch)

    test_preds_df = pd.concat(pred_dfs, ignore_index=True) if pred_dfs else \
                    pd.DataFrame(columns=["projid"] + [f"pred_{t}" for t in target_names])

    if temporary:
        try: os.remove(temp_model_path)
        except Exception: pass

    return r2_scores, training_loss_history, validation_loss_history, learning_rate_history, model


def cross_validate_lstm(model, data, n_splits, input_dim, output_dim, num_epochs, patience,
                        lr_scheduler_patience, lr_factor, hidden_size, num_layers,
                        learning_rate, batch_size, seed, dropout_rate,
                        train_ids, df_all, feature_columns, target_columns,
                        bidirectional=True, device="cpu"):

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_r2_scores = []
    fold_pred_dfs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        train_data = [data[i] for i in train_idx]
        test_data  = [data[i] for i in test_idx]
        test_ids   = [train_ids[i] for i in test_idx]   # obtain the test projid of this fold

        # train/validate
        (fold_r2, _, _, _, best_model) = train_and_evaluate_model(
            model = model,
            train_data=train_data,
            test_data=test_data,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            test_size=0.2,
            random_state=fold,
            seed=seed,
            temporary=True,
            dropout_rate=dropout_rate,
            target_names=target_columns
        )
        all_r2_scores.append(fold_r2)

        # --- NEW: Use the best weights from this fold to run full-timestep inference on test_ids and collect predictions ---
        df_fold_pred = infer_all_timesteps_df(
            trained_model=best_model,
            df_all=df_all,
            test_ids=test_ids,
            feature_cols=feature_columns,
            target_cols=target_columns,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            device=device
        )
        df_fold_pred["cv_fold"] = fold + 1
        fold_pred_dfs.append(df_fold_pred)

    all_r2_scores = np.array(all_r2_scores)
    mean_r2_scores = all_r2_scores.mean(axis=0)

    # Merge full-timestep OOF predictions from all 5 folds
    oof_df = pd.concat(fold_pred_dfs, ignore_index=True) if fold_pred_dfs else \
             pd.DataFrame(columns=["projid","fu_year"]+[f"pred_{t}" for t in target_columns])

    return mean_r2_scores, oof_df




def select_lstm_hyperparameters(model, train_sequences, feature_columns, target_columns, 
                                hyperparameter_grid, seed=1217, n_splits=5, num_epochs=100,
                                patience=10, lr_scheduler_patience=5, lr_factor=0.5,
                                train_ids=None, df_all=None, device="cpu", bidirectional=True):

    combos = list(itertools.product(
        hyperparameter_grid['hidden_size'],
        hyperparameter_grid['num_layers'],
        hyperparameter_grid['learning_rate'],
        hyperparameter_grid['batch_size'],
        hyperparameter_grid['dropout_rate']
    ))

    results = []
    best_score = -1e9
    best_oof_df = None

    for (hidden_size, num_layers, learning_rate, batch_size, dropout_rate) in combos:
        mean_r2, oof_df = cross_validate_lstm(
            model = model,
            data=train_sequences,
            n_splits=n_splits,
            input_dim=len(feature_columns),
            output_dim=len(target_columns),
            num_epochs=num_epochs,
            patience=patience,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
            dropout_rate=dropout_rate,
            train_ids=train_ids,              # <-- NEW
            df_all=df_all,                    # <-- NEW
            feature_columns=feature_columns,  # <-- NEW
            target_columns=target_columns,    # <-- NEW
            bidirectional=bidirectional,
            device=device
        )

        row = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate
        }
        # Fill in the mean R² for each target column
        for i, tgt in enumerate(target_columns):
            row[tgt] = float(mean_r2[i])
        results.append(row)

        # Use the overall average of mean R² across all targets as the selection metric
        # (you could also select based on a specific target column instead)
        combo_score = float(np.mean(mean_r2))
        if combo_score > best_score:
            best_score = combo_score
            best_oof_df = oof_df.copy()

    results_df = pd.DataFrame(results)
    return results_df, best_oof_df

