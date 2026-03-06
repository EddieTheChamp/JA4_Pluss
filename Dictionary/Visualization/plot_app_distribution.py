import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Create Dictionary", "correlated_ja4_db_large.json"))

def plot_app_distribution():
    print(f"Loading dataset from: {DATASET_FILE}")
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Drop rows without an application defined
    df = df.dropna(subset=["application"])
    
    # Get value counts for applications
    app_counts = df["application"].value_counts().reset_index()
    app_counts.columns = ["Application", "Count"]
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Create the bar plot (horizontal for readability of app names)
    ax = sns.barplot(x="Count", y="Application", data=app_counts, palette="viridis")
    
    # Add data labels
    for i, v in enumerate(app_counts["Count"]):
        ax.text(v + 1, i, str(v), color='black', va='center')
        
    plt.title(f"Application Distribution in Original Dataset (Total: {len(df)} records)", fontsize=16, pad=15)
    plt.xlabel("Number of Occurrences", fontsize=12)
    plt.ylabel("Application", fontsize=12)
    plt.tight_layout()
    
    out_file = os.path.join(SCRIPT_DIR, "app_distribution_large.png")
    plt.savefig(out_file, dpi=300)
    print(f"Graph saved to {out_file}")

if __name__ == "__main__":
    plot_app_distribution()
