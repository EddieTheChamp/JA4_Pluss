import json
import os
from typing import Dict, Optional, Any

class JA4PlusDatabase:
    """
    A dictionary matching model for JA4+ network fingerprints.
    
    This class loads a local JSON database (ja4+_db.json) and builds an EXACT MATCH index 
    based strictly on the selected experimental mode.
    """
    
    LOCAL_JSON_PATH = "ja4+_db.json"

    def __init__(self, mode: str = "ja4_only"):
        """
        Initialize the database for a specific experiment mode.
        
        Args:
            mode (str): The experimental mode. Valid options are:
                - "ja4_only" : Matches strictly using the JA4 string.
                - "ja4_ja4s" : Matches strictly using a combined JA4 and JA4S string.
                - "ja4_ja4s_ja4ts" : Matches strictly using all three components combined.
        """
        valid_modes = ["ja4_only", "ja4_ja4s", "ja4_ja4s_ja4ts"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
            
        self.mode = mode
        
        # This will hold our exact match index.
        # Key: The combined string based on 'mode' (e.g., "ja4_string" or "ja4_string|ja4s_string")
        # Value: A list of dictionaries containing Application, OS, Device, etc.
        self.index: Dict[str, list] = {}
        
    def load_database(self) -> None:
        """
        Loads the database from the local JSON file.
        Builds the internal index based on self.mode.
        """
        if not os.path.exists(self.LOCAL_JSON_PATH):
            raise FileNotFoundError(f"Database file not found: {self.LOCAL_JSON_PATH}")
            
        print(f"Loading local Database from {self.LOCAL_JSON_PATH}...")
        with open(self.LOCAL_JSON_PATH, 'r', encoding='utf-8') as f:
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
