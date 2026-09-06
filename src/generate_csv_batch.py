import os
import argparse
import pandas as pd
import numpy as np

def generate_from_dataset(source_path: str, output_path: str, n_normal: int = 15, n_fraud: int = 5):
    """Samples real rows from creditcard.csv and strips the label."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found at {source_path}")

    print(f"Loading {source_path}...")
    df = pd.read_csv(source_path)

    normal_samples = df[df["Class"] == 0].sample(n=n_normal, random_state=42)
    fraud_samples = df[df["Class"] == 1].sample(n=n_fraud, random_state=42)

    # Combine, shuffle, and drop ground-truth target
    batch_df = pd.concat([normal_samples, fraud_samples]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    batch_df = batch_df.drop(columns=["Class"])

    # Ensure correct column ordering
    ordered_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    batch_df = batch_df[ordered_cols]

    batch_df.to_csv(output_path, index=False)
    print(f"Saved real-sample batch ({len(batch_df)} rows) to: {output_path}")

def generate_synthetic(output_path: str, n_rows: int = 20, fraud_ratio: float = 0.25):
    """Generates synthetic transactions without needing the full Kaggle dataset."""
    np.random.seed(42)
    
    n_fraud = int(n_rows * fraud_ratio)
    data = {
        "Time": np.linspace(100.0, 3600.0, n_rows),
        "Amount": np.round(np.random.exponential(scale=75.0, size=n_rows) + 5.0, 2)
    }

    # Generate standard PCA signals
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(loc=0.0, scale=1.0, size=n_rows)

    df_synth = pd.DataFrame(data)

    # Inject fraud anomalies into specific rows
    fraud_indices = np.random.choice(n_rows, size=n_fraud, replace=False)
    for idx in fraud_indices:
        df_synth.loc[idx, "V1"] = -3.2
        df_synth.loc[idx, "V2"] = 4.1
        df_synth.loc[idx, "V3"] = -5.5
        df_synth.loc[idx, "V4"] = 4.8
        df_synth.loc[idx, "V14"] = -6.5
        df_synth.loc[idx, "Amount"] = 450.00

    ordered_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df_synth = df_synth[ordered_cols]

    df_synth.to_csv(output_path, index=False)
    print(f"Saved synthetic batch ({len(df_synth)} rows with ~{n_fraud} fraud cases) to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a transaction batch CSV for testing.")
    parser.add_argument("--mode", choices=["real", "synthetic"], default="real", help="Source type: 'real' uses creditcard.csv, 'synthetic' creates mock data")
    parser.add_argument("--output", default="batch_test_transactions.csv", help="Destination file path")
    parser.add_argument("--count", type=int, default=20, help="Total number of records to generate")
    args = parser.parse_args()

    data_file = os.path.join("data", "creditcard.csv")

    if args.mode == "real" and os.path.exists(data_file):
        n_fraud = max(1, int(args.count * 0.25))
        n_normal = args.count - n_fraud
        generate_from_dataset(data_file, args.output, n_normal=n_normal, n_fraud=n_fraud)
    else:
        generate_synthetic(args.output, n_rows=args.count)