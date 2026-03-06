import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

def load_and_preprocess_data(db_path="../Dictionary/ja4+_db.json"):
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    
    # "well-known crawler identifiers such as Googlebot, Bingbot, or LinkedInBot 
    # were classified as 'good bots' and excluded from the training data"
    good_bots = ["googlebot", "bingbot", "linkedinbot", "google bot", "bing bot"]

    for d in data:
        ja4 = d.get("ja4_fingerprint")
        if not isinstance(ja4, str) or "_" not in ja4:
            continue
            
        parts = ja4.split("_")
        if len(parts) != 3:
            continue
            
        ja4_a, ja4_b, ja4_c = parts
        if len(ja4_a) != 10:
            continue
            
        app = str(d.get("application") or "").lower()
        user_agent = str(d.get("user_agent_string") or "").lower()
        
        # Check for good bots in application or user_agent_string
        is_good_bot = any(gb in app or gb in user_agent for gb in good_bots)
        
        if is_good_bot:
            continue # Exclude good bots completely from dataset
        
        # The paper classified remaining entries as 'bad bots' if 'bot' was present. 
        # Our raw DB metadata stores this mainly in the user_agent_string.
        is_bad_bot = "bot" in app or "bot" in user_agent
            
        # 0 = Benign, 1 = Bad Bot
        label = 1 if is_bad_bot else 0
            
        row = {
            "ja4": ja4,
            "ja4_a_protocol": ja4_a[0],
            "ja4_a_tls": ja4_a[1:3],
            "ja4_a_sni": ja4_a[3],
            "ja4_a_cipher_cnt": ja4_a[4:6],
            "ja4_a_ext_cnt": ja4_a[6:8],
            "ja4_a_alpn": ja4_a[8:10],
            "ja4_b": ja4_b,
            "ja4_c": ja4_c,
            "label": label
        }
        records.append(row)
        
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} valid JA4 samples (Good bots excluded).")
    if len(df) > 0:
        print(f"Bad Bots (1): {(df['label'] == 1).sum()}, Benign (0): {(df['label'] == 0).sum()}")
    
    return df

def prepare_datasets(df, random_seed=42):
    # Vectorize categorical features using OneHotEncoder
    features = [
        "ja4_a_protocol", "ja4_a_tls", "ja4_a_sni", 
        "ja4_a_cipher_cnt", "ja4_a_ext_cnt", "ja4_a_alpn",
        "ja4_b", "ja4_c"
    ]
    
    df[features] = df[features].fillna("missing")
    
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_df = pd.DataFrame(encoder.fit_transform(df[features]), columns=features)
    y = df["label"]
    
    # Train-test split (80/20) with a fixed random seed
    stratify = y if sum(y) > 1 and sum(y) < len(y)-1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=random_seed, stratify=stratify
    )
    
    print("Data preparation complete.")
    print(f"Bot vs Benign Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return {
        "bot": (X_train, X_test, y_train, y_test),
        "encoder": encoder,
        "feature_names": features
    }

if __name__ == "__main__":
    df = load_and_preprocess_data()
    if len(df) > 0:
        prepare_datasets(df)
