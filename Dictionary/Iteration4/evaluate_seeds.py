import json
import pandas as pd
from data_loader import load_and_preprocess_data, prepare_datasets
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_seeds():
    df = load_and_preprocess_data("../Dictionary/ja4+_db.json")
    if len(df) == 0:
        print("No valid data loaded.")
        return
        
    seeds = [42, 123, 777]
    results = []

    for seed in seeds:
        print(f"\n==========================================")
        print(f"Running evaluation with random_seed={seed}")
        print(f"==========================================")
        
        datasets = prepare_datasets(df, random_seed=seed)
        X_train, X_test, y_train, y_test = datasets["bot"]
        
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1),
            "CatBoost": CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=seed, thread_count=-1)
        }
        
        for name, model in models.items():
            print(f"Training {name} (Seed {seed})...")
            model.fit(X_train, y_train)
            
            y_pred_te = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred_te)
            prec = precision_score(y_test, y_pred_te, zero_division=0)
            rec = recall_score(y_test, y_pred_te, zero_division=0)
            f1 = f1_score(y_test, y_pred_te, zero_division=0)
            
            results.append({
                "Seed": seed,
                "Model": name,
                "Test Acc": acc,
                "Test Prec": prec,
                "Test Recall": rec,
                "Test F1": f1
            })
            
    print("\n\n=== CROSS-SEED METRICS COMPARISON (TEST DATA) ===")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n=== AVERAGE METRICS BY MODEL ===")
    avg_df = results_df.groupby("Model").mean().drop(columns=["Seed"])
    print(avg_df.to_string())

if __name__ == "__main__":
    evaluate_seeds()
