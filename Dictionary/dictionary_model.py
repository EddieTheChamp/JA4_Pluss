import json
import os
import argparse
import pandas as pd
from typing import Dict, Optional, Any
from sklearn.model_selection import train_test_split

class JA4PlusDatabase:
    """
    A dictionary matching model for JA4+ network fingerprints.
    
    This class loads a local JSON database and builds an EXACT MATCH index 
    based strictly on the selected experimental mode.
    """

    def __init__(self, mode: str = "ja4_only", db_path: str = "ja4+_db.json"):
        """
        Initialize the database for a specific experiment mode.
        
        Args:
            mode (str): The experimental mode. This dictates how strict the matching is.
                - "ja4_only" : Matches strictly using ONLY the JA4 string (TCP/TLS properties).
                               This is less specific but finds more generic matches.
                - "ja4_ja4s" : Matches strictly using a combined JA4 (Client) AND JA4S (Server) string.
                               This is highly specific to a particular client-server interaction.
                - "ja4_ja4s_ja4ts" : Matches strictly using all three components combined.
                               This is extremely rigid and will only match exact replicas of traffic.
            db_path (str): The path to the JSON database file to load into memory.
        """
        valid_modes = ["ja4_only", "ja4_ja4s", "ja4_ja4s_ja4ts"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
            
        self.mode = mode
        self.db_path = db_path
        
        # This will hold our exact match index.
        # Key: The combined string based on 'mode' (e.g., "ja4_string" or "ja4_string|ja4s_string")
        # Value: A list of dictionaries containing Application, OS, Device, etc.
        self.index: Dict[str, list] = {}
        
    def load_database(self) -> None:
        """
        Loads the database from the local JSON file.
        Builds the internal index based on self.mode.
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
            
        print(f"Loading local Database from {self.db_path}...")
        with open(self.db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for row in data:
            # Safely handle potential None values by replacing them with empty strings
            ja4 = (row.get("ja4_fingerprint") or "").strip()
            ja4s = (row.get("ja4s_fingerprint") or "").strip()
            
            # Use ja4ts_fingerprint if available, otherwise fallback to ja4t_fingerprint
            ja4ts = (row.get("ja4ts_fingerprint") or row.get("ja4t_fingerprint") or "").strip()
            
            app = (row.get("application") or "").strip()
            device = (row.get("device") or "").strip()
            library = (row.get("library") or "").strip()
            user_agent = (row.get("user_agent_string") or "").strip()
                
            # --- 1. Construct the Index Key based on the MODE ---
            index_key = None
            
            if self.mode == "ja4_only" and ja4:
                index_key = ja4
            elif self.mode == "ja4_ja4s" and (ja4 or ja4s):
                # Glue them together to make a highly specific key
                index_key = f"{ja4}|{ja4s}"
            elif self.mode == "ja4_ja4s_ja4ts" and (ja4 or ja4s or ja4ts):
                # Glue all three together
                index_key = f"{ja4}|{ja4s}|{ja4ts}"
            
            # --- 2. Save the metadata if a valid key was constructed ---
            if index_key:
                if index_key not in self.index:
                    self.index[index_key] = []
                    
                # Append the label metadata
                self.index[index_key].append({
                    "Application": app,
                    "Library": library,
                    "Device": device,
                    "OS": (row.get("os") or "").strip(),
                    "UserAgent": user_agent,
                    "Notes": (row.get("notes") or "").strip(),
                })
                
                # Note: Overwriting previous identical entries is acceptable for this basic usage.

        print("Database loaded successfully.")
                
    def predict(self, ja4: str = "", ja4s: str = "", ja4ts: str = "") -> Optional[Dict[str, Any]]:
        """
        Predicts the application based on the provided fingerprint components. 
        It strictly enforces the current mode.
        """
        target_key = None
        
        if self.mode == "ja4_only":
            if not ja4:
                return {"result": "unknown", "reason": "Missing required JA4 string"}
            target_key = ja4
            
        elif self.mode == "ja4_ja4s":
            if not ja4 and not ja4s:
                return {"result": "unknown", "reason": "Missing required JA4 or JA4S strings for this mode"}
            target_key = f"{ja4}|{ja4s}"
            
        elif self.mode == "ja4_ja4s_ja4ts":
            if not ja4 and not ja4s and not ja4ts:
                return {"result": "unknown", "reason": "Missing required strings for this mode"}
            target_key = f"{ja4}|{ja4s}|{ja4ts}"
            
        # Perform Exact Match Dictionary Lookup (O(1) time complexity)
        if target_key in self.index:
            all_matches = self.index[target_key]
            
            # Aggregate the frequency of unique metadata combinations
            frequency_map = {}
            for match in all_matches:
                # Convert the dictionary match to a hashable tuple to group identical rows
                match_tuple = tuple(sorted(match.items()))
                if match_tuple not in frequency_map:
                    frequency_map[match_tuple] = 0
                frequency_map[match_tuple] += 1
                
            # Sort by frequency descending
            sorted_matches = sorted(frequency_map.items(), key=lambda item: item[1], reverse=True)
            
            results = []
            for item, count in sorted_matches[:5]:
                match_dict = dict(item)
                match_dict["occurrences_in_database"] = count
                results.append(match_dict)
                
            additional_hidden = len(sorted_matches) - 5 if len(sorted_matches) > 5 else 0
            
            return {
                "result": "match",
                "top_matches": results,
                "additional_results_count": additional_hidden,
                "total_unique_combinations_found": len(sorted_matches)
            }
        else:
            return {"result": "unknown"}

def evaluate_test_set_to_file(dataset_path: str, db_file: str, model_name: str, mode: str, output_file: str):
    """
    Reads the full dataset, runs the identical train_test_split as other models, 
    and evaluates strictly the Test set against the provided dictionary database.
    Saves the results directly to the required prediction payload format for the graph generator.
    """
    print(f"Loading full dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df = df.dropna(subset=["application"])
    
    # Run the exact same split to isolate the unseen Test set
    y = df["application"]
    try:
        _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        print("Warning: Couldn't stratify split due to rare classes. Falling back to unstratified split.")
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Evaluating {len(test_df)} unseen test samples with Dictionary ({model_name}) using {db_file}...")
    
    db = JA4PlusDatabase(mode=mode, db_path=db_file)
    db.load_database()
    
    evaluated_results = []
    
    for _, row in test_df.iterrows():
        app = row.get("application", "Unknown")
        ja4 = row.get("ja4_fingerprint") or ""
        ja4s = row.get("ja4s_fingerprint") or ""
        ja4ts = row.get("ja4ts_fingerprint") or row.get("ja4t_fingerprint") or ""
        
        # Query the dictionary
        res = db.predict(ja4=ja4, ja4s=ja4s, ja4ts=ja4ts)
        
        # Parse the output into our standardized Graph schema
        prediction_app = "Unknown"
        top_k_list = []
        matches_count = 0
        
        if res.get("result") == "match":
            matches_count = res.get("total_unique_combinations_found", 0)
            top_matches = res.get("top_matches", [])
            
            # Extract the raw application strings to build the top_k array
            for m in top_matches:
                cand = m.get("Application")
                if cand and cand not in top_k_list:
                    top_k_list.append(cand)
                    
            if top_k_list:
                prediction_app = top_k_list[0]
                
        # Append to the final payload
        evaluated_results.append({
            "true_app": app,
            "prediction": prediction_app,
            "top_k": top_k_list,
            "matches_count": matches_count
        })

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluated_results, f, indent=4)
        
    print(f"Evaluation Complete! Saved {len(evaluated_results)} predictions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dataset on a JA4 database model.")
    parser.add_argument("--dataset_file", required=True, help="Path to the full raw dataset file")
    parser.add_argument("--db_file", required=True, help="Path to the dictionary DB to evaluate against (e.g. ja4+_db.json or egenlagd_db.json)")
    parser.add_argument("--model_name", required=True, help="Name of the model being tested (e.g., FoxIO, Egenlagd)")
    parser.add_argument("--mode", default="ja4_ja4s_ja4ts", help="The strictness mode of the dictionary matching")
    parser.add_argument("--output_file", required=True, help="Path to save the _result.json payload")
    
    args = parser.parse_args()
    evaluate_test_set_to_file(args.dataset_file, args.db_file, args.model_name, args.mode, args.output_file)
