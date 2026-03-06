import json
import argparse
import os
import sys

# Add parent directory to path so dictionary_model can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from dictionary_model import JA4PlusDatabase
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_blind_traffic(capture_file: str, db_file: str, mode: str = "ja4_ja4s_ja4ts"):
    """
    Reads a raw JSON array of blind network traffic logs and predicts 
    the active applications using the specified JA4 database.
    """
    if not os.path.exists(capture_file):
        print(f"Error: Could not find traffic capture file at {capture_file}")
        return

    if not os.path.exists(db_file):
        print(f"Error: Could not find Dictionary Database at {db_file}")
        print("Please ensure you have run 'Iteration2/iteration2.py' first to compile the DB.")
        return

    print(f"\n[JA4+ Estimator Prototype]")
    print(f"Loading Blind Traffic Capture  : {capture_file}")
    print(f"Loading Dictionary Engine      : {db_file} (Mode: {mode})")
    
    with open(capture_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Parsed {len(data)} packets in stream. Initializing prediction engine...\n")
    
    # Silence the internal database loading print statements from dictionary_model
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        db = JA4PlusDatabase(mode=mode, db_path=db_file)
        db.load_database()
    finally:
        sys.stdout = old_stdout

    results = []
    app_counts = {}
    
    print("=" * 100)
    print(f"{'INDEX':<7} | {'JA4 FINGERPRINT':<40} | {'PREDICTED SYSTEM / APPLICATION':<45}")
    print("-" * 100)

    for i, row in enumerate(data):
        # Handle variations where the key might be "ja4" or "ja4_fingerprint"
        ja4   = row.get("ja4_fingerprint") or row.get("ja4") or ""
        ja4s  = row.get("ja4s_fingerprint") or row.get("ja4s") or ""
        ja4t  = row.get("ja4t_fingerprint") or row.get("ja4t") or ""
        ja4ts = row.get("ja4ts_fingerprint") or row.get("ja4ts") or row.get("ja4t") or ""

        # Skip packets with utterly no JA4 protocol stack
        if not ja4 and not ja4s and not ja4t:
            continue

        res = db.predict(ja4=ja4, ja4s=ja4s, ja4t=ja4t, ja4ts=ja4ts)

        prediction_str = "[!] UNKNOWN TRAFFIC"
        confidence_str = ""

        if res.get("result") == "match":
            top_matches = res.get("top_matches", [])
            # Try to grab the top application
            if top_matches:
                top_app = top_matches[0].get("Application", "Unknown")
                prob = top_matches[0].get("probability_percent", 0)
                collisions = res.get("total_unique_combinations_found", 1)
                
                prediction_str = top_app
                
                if collisions == 1:
                    confidence_str = f"   (Exact Match)"
                else:
                    confidence_str = f"   (Low Confidence - {collisions} Collisions)"
                    
        # Just grab the first 38 chars of JA4 to fit in the terminal cleanly
        short_ja4 = ja4[:38] + ".." if len(ja4) > 38 else ja4
        
        # Print a rolling live log
        print(f"#{i:<5} | {short_ja4:<40} | {prediction_str}{confidence_str}")
        
        # Tally the actual predicted string for the final chart
        app_counts[prediction_str] = app_counts.get(prediction_str, 0) + 1
        
    print("=" * 100)
    print("Estimation Complete.")
    
    # --- Generate Visual Summary ---
    print("\nGenerating Visual Summary Chart...")
    df_counts = pd.DataFrame(list(app_counts.items()), columns=["Application", "Count"])
    df_counts = df_counts.sort_values(by="Count", ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Count", y="Application", data=df_counts, palette="viridis")
    
    # Add data labels
    for i, v in enumerate(df_counts["Count"]):
        ax.text(v + 0.5, i, str(v), color='black', va='center')
        
    plt.title(f"Prototype Predictions Summary (Total Packets: {len(data)})", fontsize=16, pad=15)
    plt.xlabel("Number of Packets Predicted", fontsize=12)
    plt.ylabel("Guessed Application", fontsize=12)
    plt.tight_layout()
    
    out_dir = os.path.join(PARENT_DIR, "Visualization")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "prototype_predictions.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"Prediction distribution chart saved to -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate applications from blind JA4 traffic logs.")
    parser.add_argument("--capture_file", required=True, help="Path to the blind JSON packet capture log.")
    parser.add_argument("--db_file", default=os.path.join(PARENT_DIR, "Custom_DB", "egenlagd_correlated_db.json"), 
                        help="Path to the compiled database (Default: Custom_DB/egenlagd_correlated_db.json)")
    parser.add_argument("--mode", default="ja4_ja4s_ja4ts", help="The strictness mode of the dictionary matching.")
    
    args = parser.parse_args()
    analyze_blind_traffic(args.capture_file, args.db_file, args.mode)
