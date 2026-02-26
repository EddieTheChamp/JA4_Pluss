from dictionary_model import JA4PlusDatabase

def run_tests():
    print("=========================================")
    print("Testing EXPERIMENT 1: JA4 Only Mode")
    print("=========================================")
    # Initialize in Mode 1
    db_mode1 = JA4PlusDatabase(mode="ja4_only")
    db_mode1.load_database()
    print(f"Index built with {len(db_mode1.index)} unique JA4 records.")
    
    # Test 1: Let's find Chromium Browser based on its JA4 alone
    # Found in ja4+_db.json
    chromium_ja4 = "t12d180700_4b22cbed5bed_2dae41c691ec"
    
    import json
    res1 = db_mode1.predict(ja4=chromium_ja4)
    print(f"\n[Mode 1] predicting {chromium_ja4}:")
    print(json.dumps(res1, indent=2))
    
    
    print("\n=========================================")
    print("Testing EXPERIMENT 2: JA4 + JA4S Mode")
    print("=========================================")
    # Initialize in Mode 2
    db_mode2 = JA4PlusDatabase(mode="ja4_ja4s")
    db_mode2.load_database()
    print(f"Index built with {len(db_mode2.index)} unique JA4+JA4S records.")
    
    # Test 2: Let's test Sliver Agent that has both JA4 and JA4S
    # Found in ja4+_db.json
    sliver_ja4 = "t13d190900_9dc949149365_97f8aa674fd9"
    sliver_ja4s = "t130200_1301_a56c5b993250"
    
    # What happens if we only feed JA4 in Mode 2? It will actually work now if the DB also only had JA4 for this record!
    # But for Sliver, the DB actually has a JA4S as well.
    # Therefore, predicting with JUST JA4 will fail because the target_key "t13d...|" won't match "t13d...|t130..."
    print(f"\n[Mode 2] Predicting with only JA4 (should fail to match the full entry):")
    res_fail = db_mode2.predict(ja4=sliver_ja4)
    print(json.dumps(res_fail, indent=2))
    
    # What happens if we feed exactly what we need?
    print(f"\n[Mode 2] Predicting with JA4 and JA4S:")
    res_success = db_mode2.predict(ja4=sliver_ja4, ja4s=sliver_ja4s)
    print(json.dumps(res_success, indent=2))
    
    
    print("\n=========================================")
    print("Testing EXPERIMENT 3: JA4 + JA4S + JA4TS Mode")
    print("=========================================")
    # Initialize in Mode 3
    db_mode3 = JA4PlusDatabase(mode="ja4_ja4s_ja4ts")
    db_mode3.load_database()
    print(f"Index built with {len(db_mode3.index)} unique JA4+JA4S+JA4TS records.")
    
    print("\n[Mode 3] Predicting with JA4 and JA4S (JA4TS missing):")
    res_success_3 = db_mode3.predict(ja4=sliver_ja4, ja4s=sliver_ja4s)
    print(json.dumps(res_success_3, indent=2))

if __name__ == "__main__":
    run_tests()
