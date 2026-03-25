"""Generate t-SNE plots of the JA4+ features with specific perplexities."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import database_lookup
import ja4_parser

_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = _ROOT / "results"
TSNE_PATH = RESULTS_DIR / "tsne_categories.png"

def prepare_data():
    train_records, test_records = database_lookup.train_test_split(seed=42, test_ratio=0.2)
    all_records = train_records + test_records
    samples = ja4_parser.records_to_training_samples(all_records)
    frame = pd.DataFrame(samples)
    
    for col in ja4_parser.CATEGORICAL_FEATURES:
        if col in frame.columns:
            frame[col] = frame[col].astype('category').cat.codes
            
    for col in ja4_parser.NUMERIC_FEATURES:
        if col in frame.columns:
            frame[col] = frame[col].fillna(0).astype(int)
            
    base_ignore = {"label", "category", "weight", "target"}
    static_features = set(ja4_parser.CATEGORICAL_FEATURES) | set(ja4_parser.NUMERIC_FEATURES)
    dynamic_cols = [c for c in frame.columns if (not c in static_features) and (not c in base_ignore)]
    
    for col in dynamic_cols:
        frame[col] = frame[col].fillna(0).astype(int)
        
    final_feature_columns = ja4_parser.CATEGORICAL_FEATURES + ja4_parser.NUMERIC_FEATURES + dynamic_cols
    X = frame[final_feature_columns]
    categories = frame["category"].fillna('unknown')
    
    return X, categories

def main():
    print("[tsne] Preparing data...")
    X, categories = prepare_data()
    X = X.fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    perplexities = [30, 50, 100, 200]
    print(f"[tsne] Running t-SNE for perplexities {perplexities} on {X_scaled.shape[0]} samples...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    unique_cats = sorted(categories.unique())
    palette = sns.color_palette('tab20', n_colors=len(unique_cats))
    
    for i, perp in enumerate(perplexities):
        print(f"  -> Perplexity={perp}...")
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=perp)
        X_tsne = tsne.fit_transform(X_scaled)
        
        sns.scatterplot(
            x=X_tsne[:, 0], 
            y=X_tsne[:, 1],
            hue=categories,
            hue_order=unique_cats,
            palette=palette,
            alpha=0.7,
            s=20,
            ax=axes[i],
            legend=False if i != 1 else True 
        )
        axes[i].set_title(f"Perplexity: {perp}")
        if i == 1:
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Categories")

    plt.suptitle("t-SNE of JA4+ Segmented Features by Category", fontsize=16)
    plt.tight_layout()
    
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    plt.savefig(TSNE_PATH, dpi=150, bbox_inches='tight')
    print(f"\n[tsne] Saved to {TSNE_PATH}")

if __name__ == "__main__":
    main()
