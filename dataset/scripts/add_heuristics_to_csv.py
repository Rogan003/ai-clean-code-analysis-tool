"""
Script to add heuristic features to the official dataset CSV files.
This will compute heuristics for each code snippet and add them as new columns.
"""
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.heuristics import method_heuristics, class_heuristics
from src.data import get_method_object, get_class_obj


def add_heuristics_to_methods():
    """Add heuristic features to methods dataset."""
    print("=" * 60)
    print("Processing METHODS dataset...")
    print("=" * 60)

    csv_path = os.path.join(project_root, "dataset", "official_dataset_methods.csv")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} methods")
    print(f"Existing columns: {list(df.columns)}")

    # Feature column names for methods (13 features)
    feature_cols = [
        "h_name_len",
        "h_name_special",
        "h_name_camel",
        "h_variable_len",
        "h_variable_special",
        "h_variable_camel",
        "h_method_length",
        "h_n_params",
        "h_comment_chars",
        "h_indent",
        "h_long_line",
        "h_cyclomatic",
        "h_returns"
    ]

    # Initialize feature columns
    for col in feature_cols:
        df[col] = None

    df['h_label'] = None

    # Process each method
    failed_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing method heuristics"):
        code = str(row['code_snippet'])

        try:
            method_obj = get_method_object(code)
            if method_obj is None:
                failed_count += 1
                continue

            h_result = method_heuristics(code, method_obj)

            # Add features to dataframe
            for i, col in enumerate(feature_cols):
                df.at[idx, col] = h_result.features[i]

            df.at[idx, 'h_label'] = h_result.label

        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            failed_count += 1
            continue

    # Drop rows where heuristics couldn't be computed
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    print(f"\nSuccessfully processed: {len(df_clean)} methods")
    print(f"Failed to process: {failed_count} methods")
    print(f"New columns: {list(df_clean.columns)}")

    # Save to new CSV
    output_path = os.path.join(project_root, "dataset", "official_dataset_methods.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df_clean


def add_heuristics_to_classes():
    """Add heuristic features to classes dataset."""
    print("\n" + "=" * 60)
    print("Processing CLASSES dataset...")
    print("=" * 60)

    csv_path = os.path.join(project_root, "dataset", "official_dataset_classes.csv")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} classes")
    print(f"Existing columns: {list(df.columns)}")

    # Feature column names for classes (9 features)
    feature_cols = [
        "h_name_len",
        "h_prop_name_len_avg",
        "h_prop_name_special_avg",
        "h_public_non_gs",
        "h_n_vars",
        "h_comment_chars",
        "h_avg_method_score",
        "h_lack_of_cohesion",
        "h_name_camel"
    ]

    # Initialize feature columns
    for col in feature_cols:
        df[col] = None

    df['h_label'] = None

    # Process each class
    failed_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing class heuristics"):
        code = str(row['code_snippet'])
        avg_method_score = row['average_method_score']

        try:
            class_obj = get_class_obj(code)
            if class_obj is None:
                failed_count += 1
                continue

            h_result = class_heuristics(code, class_obj, avg_method_score)

            # Add features to dataframe
            for i, col in enumerate(feature_cols):
                df.at[idx, col] = h_result.features[i]

            df.at[idx, 'h_label'] = h_result.label

        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            failed_count += 1
            continue

    # Drop rows where heuristics couldn't be computed
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    print(f"\nSuccessfully processed: {len(df_clean)} classes")
    print(f"Failed to process: {failed_count} classes")
    print(f"New columns: {list(df_clean.columns)}")

    # Save to new CSV
    output_path = os.path.join(project_root, "dataset", "official_dataset_classes.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df_clean


if __name__ == "__main__":
    print("Starting heuristics computation for datasets...")
    print("This may take a few minutes...\n")

    # Process methods
    df_methods = add_heuristics_to_methods()

    # Process classes
    df_classes = add_heuristics_to_classes()

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Methods dataset: {len(df_methods)} rows with {len(df_methods.columns)} columns")
    print(f"Classes dataset: {len(df_classes)} rows with {len(df_classes.columns)} columns")
