# configs/best_models.py

MODEL_PARAMS = {
    "amyloid": {
        "architecture": "BiLSTM",
        "target_columns": ['gpath', 'tangles', 'amyloid', 'niareagansc'],
        "hyperparameters": {
            "hidden_size": 32,
            "num_layers": 3,  # Note: Image shows 3 layers for amyloid, previous text said 4. Please verify.
            "learning_rate": 0.01,
            "dropout_rate": 0.2, # From image_3b1a10.png table
            "batch_size": 32     # From image_3b1a10.png table
        }
    },
    "tangles": {
        "architecture": "LSTM", # Corresponds to LSTM_1T in your table
        "target_columns": ['amyloid', 'niareagansc', 'tangles'],
        "hyperparameters": {
            "hidden_size": 16,
            "num_layers": 3,  # From image_3b1a10.png table
            "learning_rate": 0.01,
            "dropout_rate": 0.0,
            "batch_size": 64  # From image_3b1a10.png table
        }
    },
    "gpath": {
        "architecture": "LSTMReLU",
        "target_columns": ['gpath', 'tangles', 'niareagansc'],
        "hyperparameters": {
            "hidden_size": 8,
            "num_layers": 4,
            "learning_rate": 0.01,
            "dropout_rate": 0.0,
            "batch_size": 16
        }
    },
    "niareagansc": {
        "architecture": "BiLSTM",
        "target_columns": ['gpath', 'tangles', 'amyloid', 'niareagansc'],
        "hyperparameters": {
            "hidden_size": 32,
            "num_layers": 4,
            "learning_rate": 0.005,
            "dropout_rate": 0.0,
            "batch_size": 32
        }
    }
}