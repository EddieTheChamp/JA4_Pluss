import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_features():
    try:
        with open("feature_importances.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("feature_importances.json not found. Run train_models.py first.")
        return
        
    features = data["features"]
    importances = data["importances"]
    
    for task_name, models in importances.items():
        print(f"\n=== Feature Analysis for Task: {task_name.upper()} ===")
        
        for model_name, imp in models.items():
            print(f"-- {model_name} --")
            imp_dict = {f_name: i_val for f_name, i_val in zip(features, imp)}
            total = sum(imp_dict.values())
            
            if total > 0:
                imp_dict = {k: v/total for k, v in imp_dict.items()}
                
            sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
            for k, v in sorted_imp:
                print(f"  {k}: {v:.4f} ({v*100:.2f}%)")
                    
            print()
            
            # Graph visualization mimicking Dictionary/Visualization
            labels = list(imp_dict.keys())
            values = list(imp_dict.values())
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(labels))
            
            # Use a slightly complex color palette similar to the other graphs
            colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
            
            bars = plt.bar(x, [v * 100 for v in values], color=colors)
            plt.title(f"Segment Importance: {model_name} ({task_name})", fontsize=14)
            plt.ylabel("Relative Importance (%)", fontsize=12)
            plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
            plt.ylim(0, max([v * 100 for v in values]) * 1.15 if values else 100)
            
            # Annotate exact percentages over bars
            for rect in bars:
                height = rect.get_height()
                if height > 0:
                    plt.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
            plt.tight_layout()
            plt.savefig(f"modularity_{task_name}_{model_name.replace(' ', '_')}.png")
            plt.close()
            
    print("Modularity plots saved as PNGs.")

if __name__ == "__main__":
    analyze_features()
