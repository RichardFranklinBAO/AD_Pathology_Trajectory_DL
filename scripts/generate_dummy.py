import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
RAW_DIR = Path("data/raw")
DUMMY_DIR = Path("data/dummy")
NUM_ROWS = 50  # Generate 50 rows of dummy data

# Automatically create dummy directory
DUMMY_DIR.mkdir(parents=True, exist_ok=True)

def generate_random_data(df, num_rows):
    """Generate random data based on the input DataFrame structure."""
    dummy = pd.DataFrame()
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_integer_dtype(dtype):
            # Integers: Generate random integers (if ID, generate larger numbers)
            if 'id' in col.lower():
                dummy[col] = np.arange(100000, 100000 + num_rows)
            else:
                dummy[col] = np.random.randint(0, 10, size=num_rows)
                
        elif pd.api.types.is_float_dtype(dtype):
            # Floats: Generate random floats
            dummy[col] = np.random.rand(num_rows)
            
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            # Strings/Objects: Generate random categories
            dummy[col] = np.random.choice(['A', 'B', 'C', 'MAP', 'ROS'], size=num_rows)
            
        else:
            # Others: Fill with 0
            dummy[col] = 0
            
    return dummy

# --- Main Loop ---
print(f"Scanning {RAW_DIR} for datasets...")

for file_path in RAW_DIR.iterdir():
    if file_path.name.startswith('.'): continue # Skip hidden files
    
    try:
        print(f"Processing {file_path.name}...")
        
        # 1. Read raw data (read only first 5 rows to infer schema, faster)
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, nrows=5)
        elif file_path.suffix == '.tsv' or file_path.suffix == '.txt':
            df = pd.read_csv(file_path, sep='\t', nrows=5)
        elif file_path.suffix in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path, nrows=5)
        elif file_path.suffix == '.pkl':
            df = pd.read_pickle(file_path)
            if not isinstance(df, pd.DataFrame):
                print(f"  Skipping {file_path.name}: Not a DataFrame pickle")
                continue
            df = df.head(5)
        else:
            print(f"  Skipping {file_path.name}: Unsupported format")
            continue

        # 2. Generate dummy data
        dummy_df = generate_random_data(df, NUM_ROWS)
        
        # 3. Save to dummy directory (preserve original format)
        save_path = DUMMY_DIR / file_path.name
        
        if file_path.suffix == '.csv':
            dummy_df.to_csv(save_path, index=False)
        elif file_path.suffix == '.tsv' or file_path.suffix == '.txt':
            dummy_df.to_csv(save_path, sep='\t', index=False)
        elif file_path.suffix in ['.xls', '.xlsx']:
            dummy_df.to_excel(save_path, index=False)
        elif file_path.suffix == '.pkl':
            dummy_df.to_pickle(save_path)
            
        print(f"  Generated dummy for {file_path.name}")
        
    except Exception as e:
        print(f"  Failed to process {file_path.name}: {e}")

print("\nAll done! Dummy data is ready in data/dummy/")