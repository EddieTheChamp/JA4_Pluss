import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

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
    """
    Iterates over the dataframe to find valid JA4 fingerprints, parses them 
    into individual parts (protocol, ciphers, extensions, etc.), and returns 
    a cleaned dataframe of precisely the features the Random Forest will train on.
    """
    records = []
    valid_indices = []
    for idx, row in df.iterrows():
        ja4 = row.get("ja4_fingerprint")
        app = row.get("application")
        
        # Skip rows where we don't know the app or have no fingerprint
        if not ja4 or not app:
            continue
            
        parsed = parse_ja4_fingerprint(ja4)
        if parsed:
            # We add the target 'application' label so we can train on it
            parsed["application"] = app
            parsed["ja4_raw"] = ja4
            parsed["original_index"] = idx
            records.append(parsed)
            valid_indices.append(idx)
            
    return pd.DataFrame(records), valid_indices

def train_and_evaluate(dataset_path: str, output_result_path: str):
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Parse JSON into Pandas DataFrame and align exactly with the Dictionary split
    df = pd.DataFrame(data)
    df = df.dropna(subset=["application"]).copy()
    
    y = df["application"]
    try:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        print("Warning: Couldn't stratify split due to rare classes. Falling back to unstratified split.")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Split dataset: {len(train_df)} training samples, {len(test_df)} testing samples.")

    # 2. Extract features specifically from Train
    train_features_df, _ = extract_features(train_df)
    
    # Preprocess Categorical Features
    categorical_columns = ["protocol", "tls_version", "sni_indicator", "alpn", "ciphers_hash", "extensions_hash"]
    label_encoders = {}
    
    # We collect all possible categorical values to prevent un-seen label errors in test set
    all_features_df, _ = extract_features(df)
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(all_features_df[col].astype(str))
        label_encoders[col] = le
        train_features_df[col] = le.transform(train_features_df[col].astype(str))

    # Encode Target
    app_encoder = LabelEncoder()
    app_encoder.fit(df["application"].astype(str))
    
    train_features_df["target"] = app_encoder.transform(train_features_df["application"].astype(str))

    feature_cols = categorical_columns + ["num_ciphers", "num_extensions"]
    X_train = train_features_df[feature_cols]
    y_train = train_features_df["target"]

    # 3. Train RF
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)

    # 4. Evaluate on Test Set
    results_payload = []
    
    # We iterate over the exact test_df to guarantee chronological row-alignment with Dictionary
    for idx, row in test_df.iterrows():
        true_app = row["application"]
        ja4 = row.get("ja4_fingerprint")
        
        parsed = parse_ja4_fingerprint(ja4) if ja4 else None
        
        if not parsed:
            # Test sample cannot be evaluated, record as Unknown immediately
            results_payload.append({
                "true_app": true_app,
                "prediction": "Unknown",
                "top_k": [],
                "probability": 0.0,
                "matches_count": 0
            })
            continue
            
        # Manually encode row (handling unseen categoricals gracefully is tricky, but LabelEncoder fit on all earlier solves it)
        try:
            for col in categorical_columns:
                parsed[col] = label_encoders[col].transform([str(parsed[col])])[0]
                
            x_test_row = [parsed[col] for col in feature_cols]
            
            # Convert to DataFrame with matching column names to suppress Sklearn warnings
            x_test_df = pd.DataFrame([x_test_row], columns=feature_cols)
            class_probs = rf_model.predict_proba(x_test_df)[0]
            
            # Identify Top-5
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
                
            results_payload.append({
                "true_app": true_app,
                "prediction": predicted_app if matches_count > 0 else "Unknown",
                "top_k": top_k_apps if matches_count > 0 else [],
                "probability": round(confidence, 3),
                "matches_count": matches_count
            })
        except Exception as e:
            # Graceful fallback for unexpected errors (e.g. truly unseen labels)
            results_payload.append({
                "true_app": true_app,
                "prediction": "Unknown",
                "top_k": [],
                "probability": 0.0,
                "matches_count": 0
            })

    with open(output_result_path, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, indent=4)
        
    print(f"Successfully saved {len(results_payload)} evaluated samples to {output_result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate a Random Forest on parameterized JA4 fingerprints.")
    parser.add_argument("--dataset_file", required=True, help="Path to the main JSON dataset file to split and train on")
    parser.add_argument("--output_file", default="Random_Forest_result.json", help="Path to save the payload")
    
    args = parser.parse_args()
    train_and_evaluate(args.dataset_file, args.output_file)
