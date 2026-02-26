import json
import random

apps = [
    "Chromium Browser", "Mozilla Firefox", "Microsoft Edge", "Spotify", 
    "Discord", "Slack", "Microsoft Teams", "Zoom", 
    "Dropbox", "OneDrive"
]

def generate_raw_test_data(filepath: str, num_samples: int = 1000):
    """
    Generates a mock dataset representing the raw test samples the models will evaluate.
    """
    results = []
    
    app_fingerprints = {}
    for app in apps:
        app_fingerprints[app] = [f"t13d{random.randint(1000,9999)}_{app[:4]}_{random.randint(1000,9999)}" for _ in range(random.randint(1, 5))]
        
    shared_fp = "t13d1000_shar_1234"
    app_fingerprints["Chromium Browser"].append(shared_fp)
    app_fingerprints["Microsoft Edge"].append(shared_fp)
        
    for _ in range(num_samples):
        true_app = random.choice(apps)
        fp = random.choice(app_fingerprints[true_app])
        
        # Format explicitly requested by user
        results.append({
            "application": true_app,
            "ja4_fingerprint": fp,
            "ja4s_fingerprint": None,
            "ja4t_fingerprint": None,
            "ja4ts_fingerprint": None,
        })
        
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def generate_mock_results_for_model(raw_filepath: str, model_name: str, out_filepath: str):
    """
    Simulates a model evaluating the raw data and saving its specific _result.json
    """
    with open(raw_filepath, 'r') as f:
        raw_data = json.load(f)
        
    evaluated_results = []
    
    for row in raw_data:
        true_app = row["application"]
        
        if model_name == "FoxIO":
            pred = true_app if random.random() > 0.3 else random.choice(apps)
            top_k = [pred]
            while len(top_k) < 5:
                cand = random.choice(apps)
                if cand not in top_k:
                    top_k.append(cand)
            if random.random() < 0.1:
                pred = "Unknown"
                top_k = []
                matches = 0
            else:
                matches = 2 if row["ja4_fingerprint"] == "t13d1000_shar_1234" else 1
                
            evaluated_results.append({
                "true_app": true_app,
                "prediction": pred,
                "top_k": top_k,
                "matches_count": matches
            })
            
        elif model_name == "Egenlagd":
            pred = true_app if random.random() > 0.1 else random.choice(apps)
            top_k = [pred]
            while len(top_k) < 3:
                cand = random.choice(apps)
                if cand not in top_k:
                    top_k.append(cand)
            if random.random() < 0.05:
                pred = "Unknown"
                top_k = []
                matches = 0
            else:
                matches = 2 if row["ja4_fingerprint"] == "t13d1000_shar_1234" else 1
                
            evaluated_results.append({
                "true_app": true_app,
                "prediction": pred,
                "top_k": top_k,
                "matches_count": matches
            })
            
        elif model_name == "Random_Forest":
            pred = true_app if random.random() > 0.05 else random.choice(apps)
            top_k = [pred]
            while len(top_k) < 5:
                cand = random.choice(apps)
                if cand not in top_k:
                    top_k.append(cand)
                    
            prob = random.uniform(0.3, 0.99)
            if prob > 0.6:
                matches = 1
            elif prob > 0.4:
                matches = 2
            else:
                matches = 0
                
            evaluated_results.append({
                "true_app": true_app,
                "prediction": pred if matches > 0 else "Unknown",
                "top_k": top_k if matches > 0 else [],
                "probability": round(prob, 2),
                "matches_count": matches
            })

    with open(out_filepath, 'w') as f:
        json.dump(evaluated_results, f, indent=4)


if __name__ == "__main__":
    # 1. Generate the raw test queries that the models will evaluate
    generate_raw_test_data("raw_test_samples.json", 1000)
    print("Created raw_test_samples.json (Format provided by user)")
    
    # 2. Simulate each of the 3 models doing their evaluation and saving their own isolated file
    generate_mock_results_for_model("raw_test_samples.json", "FoxIO", "Results/dictionary_FoxIO_result.json")
    print("Created Results/dictionary_FoxIO_result.json")
    
    generate_mock_results_for_model("raw_test_samples.json", "Egenlagd", "Results/dictionary_Egenlagd_result.json")
    print("Created Results/dictionary_Egenlagd_result.json")
    
    generate_mock_results_for_model("raw_test_samples.json", "Random_Forest", "Results/Random_Forest_result.json")
    print("Created Results/Random_Forest_result.json")
