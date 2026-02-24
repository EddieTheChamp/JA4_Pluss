import json
import random
import copy

apps = [
    "Google Chrome", "Mozilla Firefox", "Microsoft Edge", "Spotify", 
    "Discord", "Slack", "Microsoft Teams", "Zoom", 
    "Dropbox", "OneDrive"
]

def generate_mock_results(filepath: str, num_samples: int = 1000):
    results = []
    
    # Assign some random JA4 fingerprints to apps
    app_fingerprints = {}
    for app in apps:
        # Each app has 1 to 5 fingerprints
        app_fingerprints[app] = [f"t13d{random.randint(1000,9999)}_{app[:4]}_{random.randint(1000,9999)}" for _ in range(random.randint(1, 5))]
        
    # Some apps might share fingerprints (collisions)
    shared_fp = "t13d1000_shar_1234"
    app_fingerprints["Google Chrome"].append(shared_fp)
    app_fingerprints["Microsoft Edge"].append(shared_fp)
        
    for _ in range(num_samples):
        true_app = random.choice(apps)
        fp = random.choice(app_fingerprints[true_app])
        
        # FoxIO Dictionary (Lower accuracy, more collisions)
        foxio_pred = true_app if random.random() > 0.3 else random.choice(apps)
        foxio_top_k = [foxio_pred]
        while len(foxio_top_k) < 5:
            cand = random.choice(apps)
            if cand not in foxio_top_k:
                foxio_top_k.append(cand)
        # 10% chance of being "Unknown"
        if random.random() < 0.1:
            foxio_pred = "Unknown"
            foxio_top_k = []
            foxio_matches = 0
        else:
            # Matches count: if it's the shared FP, it's 2, else 1
            foxio_matches = 2 if fp == shared_fp else 1
            
        # Custom Dictionary (Higher accuracy, fewer collisions)
        custom_pred = true_app if random.random() > 0.1 else random.choice(apps)
        custom_top_k = [custom_pred]
        while len(custom_top_k) < 3:
            cand = random.choice(apps)
            if cand not in custom_top_k:
                custom_top_k.append(cand)
        if random.random() < 0.05:
            custom_pred = "Unknown"
            custom_top_k = []
            custom_matches = 0
        else:
            custom_matches = 2 if fp == shared_fp else 1
            
        # Random Forest (High accuracy, predict_proba gives Top-K, no intrinsic "collisions" but we can simulate confidence as matches_count for identical schema)
        rf_pred = true_app if random.random() > 0.05 else random.choice(apps)
        rf_top_k = [rf_pred]
        while len(rf_top_k) < 5:
            cand = random.choice(apps)
            if cand not in rf_top_k:
                rf_top_k.append(cand)
        
        # We can map RF confidence to the "Collision" matrix idea: 
        # 1 match (Unique) = High Confidence, 2 matches (Collision) = Low Confidence/split decision, 0 (Unknown) = Prediction below threshold
        rf_prob = random.uniform(0.3, 0.99)
        if rf_prob > 0.6:
            rf_matches = 1 # Clear winner
        elif rf_prob > 0.4:
            rf_matches = 2 # Ambiguous between classes
        else:
            rf_matches = 0 # Unknown/Dropped
        
        results.append({
            "true_app": true_app,
            "ja4_fingerprint": fp,
            "models": {
                "foxio": {
                    "prediction": foxio_pred,
                    "top_k": foxio_top_k,
                    "matches_count": foxio_matches
                },
                "egenlagd": {
                    "prediction": custom_pred,
                    "top_k": custom_top_k,
                    "matches_count": custom_matches
                },
                "random_forest": {
                    "prediction": rf_pred if rf_matches > 0 else "Unknown",
                    "top_k": rf_top_k if rf_matches > 0 else [],
                    "probability": round(rf_prob, 2),
                    "matches_count": rf_matches
                }
            }
        })
        
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
        

if __name__ == "__main__":
    generate_mock_results("mock_results.json", 1000)
    print("Created mock_results.json")
