import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from data_loader import load_and_preprocess_data, prepare_datasets

def plot_correlation_matrix():
    print("Loading data...")
    df = load_and_preprocess_data("../Dictionary/ja4+_db.json")
    if len(df) == 0:
        print("No valid data loaded.")
        return
        
    print("Preparing datasets and encoding features...")
    datasets = prepare_datasets(df)
    
    # We combine Train and Test to see the correlation across the whole dataset
    X_train, X_test, y_train, y_test = datasets["bot"]
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    
    # Create a full dataframe with the label
    df_encoded = X_full.copy()
    df_encoded["label_is_bot"] = y_full
    
    print("Calculating correlation matrix...")
    corr_matrix = df_encoded.corr()
    
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    
    # Create a custom diverging palette
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, fmt=".2f", annot_kws={"size": 10})
                
    plt.title("JA4 Feature Correlation Matrix (Bot vs Benign)", fontsize=16, pad=20)
    plt.tight_layout()
    
    out_file = "ja4_correlation_matrix.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved correlation matrix to {out_file}")

if __name__ == "__main__":
    plot_correlation_matrix()
