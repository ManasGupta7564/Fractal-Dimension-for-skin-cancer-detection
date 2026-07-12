import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

#TO run this file python3 Booleanize.py {Path to target file} --bins {By default = 5} --output_dir {Where to save?}

def quantile_binning(data, n_bins=4):
    binned_data = []
    for column in data.columns:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        column_binned = discretizer.fit_transform(data[[column]]).astype(int)
        binned_data.append(column_binned)
    binned_data = np.hstack(binned_data)
    return pd.DataFrame(binned_data, columns=data.columns)


def booleanize(binned_data):
    booleanized_data = []
    column_names = []
    for column in binned_data.columns:
        unique_bins = sorted(binned_data[column].unique())
        for bin_value in unique_bins:
            boolean_column = (binned_data[column] == bin_value).astype(int)
            booleanized_data.append(boolean_column)
            column_names.append(f"{column}_bin_{bin_value}")
    booleanized_data = np.column_stack(booleanized_data)
    return pd.DataFrame(booleanized_data, columns=column_names)


def booleanize_array(num_array, n_bins):
    num_array = pd.DataFrame(num_array)
    num_bin = quantile_binning(num_array, n_bins)
    num_final = booleanize(num_bin)
    return num_final.to_numpy()


def main():
    parser = argparse.ArgumentParser(description="Booleanize an .npy file and save the result.")
    parser.add_argument("input_file", help="Path to input .npy file (e.g., X_train.npy)")
    parser.add_argument("--bins", type=int, default=5, help="Number of bins for quantile binning (default: 5)")
    parser.add_argument("--output_dir", default="booleanized_output", help="Directory to save the booleanized file")
    args = parser.parse_args()

    input_file = args.input_file
    n_bins = args.bins
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load input .npy file
    print(f"[INFO] Loading {input_file}...")
    data = np.load(input_file, allow_pickle=True)

    # Booleanize
    print(f"[INFO] Booleanizing data with {n_bins} bins...")
    booleanized_data = booleanize_array(data, n_bins)

    # Prepare output file name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_booleanized.npy")

    # Save the result
    np.save(output_file, booleanized_data)
    print(f"[INFO] Booleanized file saved as {output_file}")


if __name__ == "__main__":
    main()
