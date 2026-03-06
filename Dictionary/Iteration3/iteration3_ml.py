"""
Iteration 3: Machine Learning (Random Forest) Evaluation Pipeline
====================================================
Pipeline:
  0. Load dataset and perform the 80/20 train/test split.
  1. Extract modular JA4 features (ja4_a breakdown + hashes).
  2. Train a Random Forest model on the TRAINING split.
  3. Evaluate all TEST samples using the trained RF model.
  4. Generate visual graphs (Confusion Matrix).

Usage:
    python Iteration3/iteration3_ml.py
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
VIZ_DIR    = os.path.join(PARENT_DIR, "Visualization")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, VIZ_DIR)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from generate_comparison_graphs import plot_confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_FILE   = os.path.join(PARENT_DIR, "..", "Create Dictionary", "correlated_ja4_db_large.json")
RESULTS_DIR    = os.path.join(PARENT_DIR, "Results")  # Iteration 2 results 
ITER3_DIR      = SCRIPT_DIR
RESULT_FILE    = os.path.join(ITER3_DIR, "Random_Forest_result.json")
MODEL_NAME     = "Random_Forest"
RANDOM_STATE   = 42


def parse_ja4_fingerprint(ja4_str):
    if not ja4_str or not isinstance(ja4_str, str) or "_" not in ja4_str:
        return None
    parts = ja4_str.split("_")
    if len(parts) != 3:
        return None
    
    # Pad to 10 characters so the substring extraction doesn't crash on mock data
    protocol_tls_alpn = parts[0].ljust(10, '0')
    
    return {
        "protocol": protocol_tls_alpn[0],
        "tls_version": protocol_tls_alpn[1:3],
        "sni_indicator": protocol_tls_alpn[3],
        "num_ciphers": int(protocol_tls_alpn[4:6]) if protocol_tls_alpn[4:6].isdigit() else 0,
        "num_extensions": int(protocol_tls_alpn[6:8]) if protocol_tls_alpn[6:8].isdigit() else 0,
        "alpn": protocol_tls_alpn[8:10],
        "ciphers_hash": parts[1],
        "extensions_hash": parts[2]
    }

def extract_features(df):
    records = []
    valid_indices = []
    for idx, row in df.iterrows():
        ja4 = row.get("ja4_fingerprint")
        app = row.get("application")
        
        if not ja4 or not app:
            continue
            
        parsed = parse_ja4_fingerprint(ja4)
        if parsed:
            parsed["application"] = app
            parsed["ja4_raw"] = ja4
            parsed["original_index"] = idx
            records.append(parsed)
            valid_indices.append(idx)
            
    return pd.DataFrame(records), valid_indices


def train_and_evaluate(dataset_path: str, output_result_path: str):
    print(f"\n[Step 0] Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df = df.dropna(subset=["application"]).copy()
    
    y = df["application"]
    try:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    except ValueError:
        print("Warning: Couldn't stratify split. Falling back to unstratified split.")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    print(f"         Split dataset: {len(train_df)} training samples (80%), {len(test_df)} testing samples (20%).")

    print("[Step 1] Extracting modular JA4 features...")
    train_features_df, _ = extract_features(train_df)
    
    categorical_columns = ["protocol", "tls_version", "sni_indicator", "alpn", "ciphers_hash", "extensions_hash"]
    label_encoders = {}
    
    all_features_df, _ = extract_features(df)
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(all_features_df[col].astype(str))
        label_encoders[col] = le
        train_features_df[col] = le.transform(train_features_df[col].astype(str))

    app_encoder = LabelEncoder()
    app_encoder.fit(df["application"].astype(str))
    
    train_features_df["target"] = app_encoder.transform(train_features_df["application"].astype(str))

    feature_cols = categorical_columns + ["num_ciphers", "num_extensions"]
    X_train = train_features_df[feature_cols]
    y_train = train_features_df["target"]

    print("[Step 2] Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    print(f"[Step 3] Evaluating {len(test_df)} test samples...")
    results_payload = []
    
    y_true_labels = []
    y_pred_labels = []

    for idx, row in test_df.iterrows():
        true_app = row["application"]
        ja4 = row.get("ja4_fingerprint")
        
        parsed = parse_ja4_fingerprint(ja4) if ja4 else None
        
        if not parsed:
            results_payload.append({
                "true_app": true_app,
                "prediction": "Unknown",
                "top_k": [],
                "probability": 0.0,
                "matches_count": 0
            })
            y_true_labels.append(true_app)
            y_pred_labels.append("Unknown")
            continue
            
        try:
            for col in categorical_columns:
                parsed[col] = label_encoders[col].transform([str(parsed[col])])[0]
                
            x_test_row = [parsed[col] for col in feature_cols]
            x_test_df = pd.DataFrame([x_test_row], columns=feature_cols)
            class_probs = rf_model.predict_proba(x_test_df)[0]
            
            top_indices = np.argsort(class_probs)[::-1][:5]
            top_k_apps = app_encoder.inverse_transform(top_indices).tolist()
            
            predicted_class_index = top_indices[0]
            predicted_app = top_k_apps[0]
            confidence = class_probs[predicted_class_index]
            
            if confidence > 0.6:
                matches_count = 1
            elif confidence > 0.3:
                matches_count = 2
            else:
                matches_count = 0
                
            prediction = predicted_app if matches_count > 0 else "Unknown"
            
            results_payload.append({
                "true_app": true_app,
                "prediction": prediction,
                "top_k": top_k_apps if matches_count > 0 else [],
                "probability": round(confidence, 3),
                "matches_count": matches_count
            })
            
            y_true_labels.append(true_app)
            y_pred_labels.append(prediction)
            
        except Exception as e:
            results_payload.append({
                "true_app": true_app,
                "prediction": "Unknown",
                "top_k": [],
                "probability": 0.0,
                "matches_count": 0
            })
            y_true_labels.append(true_app)
            y_pred_labels.append("Unknown")

    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    with open(output_result_path, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, indent=4)
        
    print(f"         Prediction JSON generated -> {output_result_path}")
    
    print("[Step 4] Generating Visual Charts...")
    try:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        plot_confusion_matrix(y_true_labels, y_pred_labels, MODEL_NAME)
        
        # Load Iteration 2 results to fuse into the comparative charts
        foxio_res = []
        foxio_path = os.path.join(RESULTS_DIR, "foxio_Egenlagd_result.json")
        if os.path.exists(foxio_path):
            with open(foxio_path, 'r', encoding='utf-8') as f:
                foxio_res = json.load(f)
                
        egenlagd_res = []
        egenlagd_path = os.path.join(RESULTS_DIR, "correlated_Egenlagd_result.json")
        if os.path.exists(egenlagd_path):
            with open(egenlagd_path, 'r', encoding='utf-8') as f:
                egenlagd_res = json.load(f)
                
        from generate_comparison_graphs import plot_collision_matrix, plot_top_k
        plot_collision_matrix(foxio_res, egenlagd_res, results_payload)
        plot_top_k(foxio_res, egenlagd_res, results_payload)
        
    finally:
        sys.stdout = old_stdout
        
    print("         Confusion Matrix and 3-Model Comparatives saved to Visualization directory.\n")
    return results_payload


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') # Silence sklearn/pandas warnings

    print("="*80)
    print("ITERATION 3 - Random Forest Classification Pipeline".center(80))
    print("="*80)
    
    train_and_evaluate(DATASET_FILE, RESULT_FILE)
    print("Done!")
