import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.manifold import TSNE
from pathlib import Path

# Fix relative import
_ROOT = Path(__file__).resolve().parent
sys.path.append(str(_ROOT))

import database_lookup
import ja4_parser

def main():
    print("[visualize_tsne] Loading database...")
    records = database_lookup.load_db()
    
    # We will only use records that have a valid category
    filtered_records = [r for r in records if r.category and r.category.lower() != "unknown"]
    print(f"[visualize_tsne] Found {len(filtered_records)} records with categories out of {len(records)} total.")
    
    if not filtered_records:
        print("[visualize_tsne] Error: No categorized records found. Exiting.")
        return

    # Generate features using the ja4_parser
    print("[visualize_tsne] Parsing records via ja4_parser...")
    
    dict_features = []
    categories = []
    
    for r in filtered_records:
        obs = ja4_parser.Observation(
            observation_id="tsne",
            ja4=r.ja4, ja4s=r.ja4s,
            ja4_string=r.ja4_string, ja4s_string=r.ja4s_string,
            ja4t=r.ja4t, ja4ts=r.ja4ts
        )
        
        # This converts the observation into a dictionary of features.
        # String fields (like protocol="t") will be One-Hot Encoded.
        # Binary fields (like cipher_XYZ=1) are inherently Multi-Hot Encoded.
        features = ja4_parser.observation_to_features(obs)
        
        # User requirement: keep the ja4_a part (prefix features) and the unhashed strings, but exclude any raw hashes.
        
        valid_keys = (
            "protocol", "tls_version", "sni_indicator", "num_ciphers", "num_extensions", "alpn",
            "cipher_", "ext_", "sig_", "ja4s_cipher_", "ja4s_ext_"
        )
        
        cleaned_features = {
            k: str(v) if isinstance(v, str) else v
            for k, v in features.items() 
            if k.startswith(valid_keys)
        }
        
        if not cleaned_features:
            continue
            
        dict_features.append(cleaned_features)
        categories.append(r.category)
        
    print(f"[visualize_tsne] Vectorizing {len(dict_features)} samples using DictVectorizer (One-Hot Encoding)...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(dict_features)
    
    print(f"[visualize_tsne] Extracted {X.shape[1]} string features. Running t-SNE...")
    
    # Tuned TSNE parameters
    # perplexity between 5 and 50 is recommended. 30 is a good default.
    # init='pca' is more stable than random initialization.
    tsne = TSNE(
        n_components=2, 
        perplexity=30.0, 
        random_state=42
    )
    
    X_2d = tsne.fit_transform(X)
    
    print("[visualize_tsne] t-SNE complete. Generating plot...")
    
    df = pd.DataFrame({
        'tsne_1': X_2d[:, 0],
        'tsne_2': X_2d[:, 1],
        'category': categories
    })
    
    # Setup the plot
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Create the scatterplot
    scatter = sns.scatterplot(
        x='tsne_1', 
        y='tsne_2', 
        hue='category', 
        palette='tab20', 
        data=df, 
        alpha=0.7, 
        edgecolor=None,
        s=30
    )
    
    plt.title('t-SNE Visualization of JA4/JA4S Signatures by Category', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Category")
    plt.tight_layout()
    
    # Save the output
    out_dir = _ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "tsne_category_fullstrings.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[visualize_tsne] Saved plot to {out_path}")

if __name__ == "__main__":
    main()
