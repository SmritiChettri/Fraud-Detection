import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

data_path = os.path.join("data", "creditcard.csv")
if not os.path.exists(data_path):
    print(f"File not found: {data_path}")
    exit(1)

print("Loading data...")
df = pd.read_csv(data_path)

# Reference: first 5,000 baseline transactions
ref_data = df.iloc[:5000]

# Production: next 5,000 transactions with simulated drift on Amount
curr_data = df.iloc[5000:10000].copy()
curr_data["Amount"] = curr_data["Amount"] * 2.2

print("\nRunning Kolmogorov-Smirnov Drift Tests (p-value threshold = 0.05):")
print("-" * 65)
print(f"{'Feature':<15} {'KS Statistic':<15} {'p-value':<15} {'Drift Detected?'}")
print("-" * 65)

drift_results = []
for col in ["Amount", "V1", "V2", "V3", "V4"]:
    stat, p_val = ks_2samp(ref_data[col], curr_data[col])
    drift = p_val < 0.05
    drift_results.append({"feature": col, "ks_stat": stat, "p_value": p_val, "drift": drift})
    status = "YES (DRIFT)" if drift else "NO"
    print(f"{col:<15} {stat:<15.4f} {p_val:<15.4e} {status}")

# Generate a standalone lightweight HTML report
os.makedirs("monitoring/reports", exist_ok=True)
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Data Drift Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 30px; background: #f8fafc; }}
        h1 {{ color: #0f172a; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 800px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background-color: #0f172a; color: white; }}
        .drift {{ color: #dc2626; font-weight: bold; }}
        .no-drift {{ color: #16a34a; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Data Drift Detection Report</h1>
    <p>Statistical Metric: <b>Two-Sample Kolmogorov-Smirnov Test</b> (&alpha; = 0.05)</p>
    <table>
        <tr>
            <th>Feature</th>
            <th>KS Statistic</th>
            <th>p-value</th>
            <th>Status</th>
        </tr>
        {''.join([f"<tr><td>{r['feature']}</td><td>{r['ks_stat']:.4f}</td><td>{r['p_value']:.4e}</td><td class='{'drift' if r['drift'] else 'no-drift'}'>{'DRIFT DETECTED' if r['drift'] else 'HEALTHY'}</td></tr>" for r in drift_results])}
    </table>
</body>
</html>
"""

report_file = os.path.join("monitoring", "reports", "data_drift_report.html")
with open(report_file, "w") as f:
    f.write(html_content)

print("\n--> SUCCESS! Open your report:")
print(os.path.abspath(report_file))