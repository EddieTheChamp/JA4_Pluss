import json
import argparse
import sys
import os

# Add parent directory to path to import dictionary_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dictionary_model import JA4PlusDatabase

def analyze_raw_capture(capture_file: str, db_file: str):
    """
    Reads a raw JSON array of JA4 traffic fingerprints and evaluates them
    against the dictionary DB. Instead of plotting accuracy, it summarizes 
    what the database thinks the traffic is based on available metadata,
    and presents it in a clean format.
    """
    print(f"Loading raw capture from {capture_file}...")
    with open(capture_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} packets.")
    
    print(f"Loading FoxIO Database from {db_file}...\n")
    
    modes_to_test = ["ja4_only", "ja4s_only", "ja4t_only", "ja4ts_only", "ja4_ja4s", "ja4_ja4s_ja4ts"]
    
    # Store aggregated stats to print side-by-side in a table later
    all_mode_stats = {}
    
    for current_mode in modes_to_test:
        print("\n" + "="*80)
        print(f"[{current_mode.upper()}]".center(80))
        print("="*80)
        
        db = JA4PlusDatabase(mode=current_mode, db_path=db_file)
        db.load_database()
        
        stats = {
            "unique_matches": 0,
            "collisions": 0,
            "unknowns": 0
        }
        
        valid_packets_in_mode = 0
        
        unique_fingerprints_in_capture = set()
        unique_fingerprints_matched = set()
        
        metadata_counter = {}
    
        for row in data:
            ja4 = row.get("ja4") or "" # Note the key in this raw json is "ja4", not "ja4_fingerprint"
            ja4s = row.get("ja4s") or ""
            ja4t = row.get("ja4t") or ""
            ja4ts = row.get("ja4ts") or row.get("ja4t") or ""
            # Determine the key used for uniqueness counting based on mode
            fp_key = None
            if current_mode == "ja4_only": fp_key = ja4
            elif current_mode == "ja4s_only": fp_key = ja4s
            elif current_mode == "ja4t_only": fp_key = ja4t
            elif current_mode == "ja4ts_only": fp_key = ja4ts
            elif current_mode == "ja4_ja4s": fp_key = f"{ja4}|{ja4s}"
            elif current_mode == "ja4_ja4s_ja4ts": fp_key = f"{ja4}|{ja4s}|{ja4ts}"
            
            if fp_key and fp_key != "|" and fp_key != "||":
                unique_fingerprints_in_capture.add(fp_key)
                valid_packets_in_mode += 1
            else:
                # If this mode requires strings that don't exist in this packet, 
                # we skip it so it doesn't skew the "Unknown" tally.
                continue
            
            # Query the dictionary
            res = db.predict(ja4=ja4, ja4s=ja4s, ja4t=ja4t, ja4ts=ja4ts)
            
            if res.get("result") == "match":
                if fp_key:
                    unique_fingerprints_matched.add(fp_key)
                    
                matches_count = res.get("total_unique_combinations_found", 0)
                
                if matches_count == 1:
                    stats["unique_matches"] += 1
                else:
                    stats["collisions"] += 1
                    
                top_matches = res.get("top_matches", [])
                for m in top_matches:
                    # Extract whatever metadata is available 
                    app = m.get("Application") or "N/A"
                    device = m.get("Device") or "N/A"
                    os_val = m.get("OS") or "N/A"
                    user_agent = m.get("UserAgent") or "N/A"
                    
                    # Create a signature summarizing this match
                    signature = f"App: {app} | OS: {os_val} | Device: {device}"
                    
                    if signature not in metadata_counter:
                        metadata_counter[signature] = 0
                    metadata_counter[signature] += 1
                    
            else:
                stats["unknowns"] += 1

        total = valid_packets_in_mode
        
        if total == 0:
            all_mode_stats[current_mode.upper()] = {
                "Valid Packets": 0,
                "Unique Matches": "0 (0.0%)",
                "Collisions": "0 (0.0%)",
                "Unknowns": "0 (0.0%)",
                "Unique FPs in Capture": 0,
                "Unique FPs Matched": 0
            }
        else:
            # Save to aggregated dict
            all_mode_stats[current_mode.upper()] = {
                "Valid Packets": total,
                "Unique Matches": f"{stats['unique_matches']} ({(stats['unique_matches']/total*100):.1f}%)",
                "Collisions": f"{stats['collisions']} ({(stats['collisions']/total*100):.1f}%)",
                "Unknowns": f"{stats['unknowns']} ({(stats['unknowns']/total*100):.1f}%)",
                "Unique FPs in Capture": len(unique_fingerprints_in_capture),
                "Unique FPs Matched": len(unique_fingerprints_matched)
            }
        
        print("\n--- TOP PREDICTED METADATA SIGNATURES FOR THIS MODE ---")
        if not metadata_counter:
            print("  No matches were found in the database for any packets.")
        else:
            # Sort by most frequent
            sorted_metadata = sorted(metadata_counter.items(), key=lambda item: item[1], reverse=True)
            
            for sig, count in sorted_metadata[:10]: # Print top 10
                percentage = (count / total) * 100
                print(f"  - {count:>2} hits ({percentage:>5.1f}%) -> {sig}")
        print("\n")

    # ----- Print Final Summary Tables ----- #
    print("="*80)
    print("Iteration 1 Result Summary: Packet Statistics".center(80))
    print("="*80)
    
    # Print Headers
    header1 = f"{'MODE':<18} | {'UNIQUE MATCHES':<18} | {'COLLISIONS':<18} | {'UNKNOWNS':<18}"
    print(header1)
    print("-" * len(header1))
    
    # Print Rows
    for mode, mode_stats in all_mode_stats.items():
        print(f"{mode:<18} | {mode_stats['Unique Matches']:<18} | {mode_stats['Collisions']:<18} | {mode_stats['Unknowns']:<18}")
        
    print("\n\n")

    print("="*60)
    print("Iteration 1 Result Summary: Unique Fingerprints".center(60))
    print("="*60)
    
    # Print Headers
    header2 = f"{'MODE':<18} | {'UNIQUE FPs (TOTAL)':<18} | {'UNIQUE FPs (HIT)':<18}"
    print(header2)
    print("-" * len(header2))
    
    # Print Rows
    for mode, mode_stats in all_mode_stats.items():
        print(f"{mode:<18} | {mode_stats['Unique FPs in Capture']:<18} | {mode_stats['Unique FPs Matched']:<18}")
        
    print("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a raw JA4 capture against the FoxIO DB.")
    parser.add_argument("--capture_file", default="firefox.json", help="Path to the raw capture JSON")
    parser.add_argument("--db_file", default="../ja4+_db.json", help="Path to the FoxIO DB")
    
    args = parser.parse_args()
    analyze_raw_capture(args.capture_file, args.db_file)
