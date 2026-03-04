import pandas as pd
import json
import os
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def correlate_traffic_best_effort(csv_file, json_file, output_file, time_delta_seconds=2, domain_aware=True):
    print("[*] Loading data...")
    df_csv = pd.read_csv(csv_file)
    
    with open(json_file, 'r') as f:
        ja4_data = json.load(f)
    df_json = pd.DataFrame(ja4_data)

    # 1. Clean the 'app' column
    df_csv['app'] = df_csv['app'].apply(
        lambda x: os.path.basename(str(x).replace('\\', '/')) if pd.notna(x) else None
    )

    # 2. Standardize timestamps and sort (Required)
    df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])
    df_json['timestamp'] = pd.to_datetime(df_json['A'])
    df_csv = df_csv.sort_values('timestamp')
    df_json = df_json.sort_values('timestamp')

    # 3. Clean up the domain columns to handle pure missing values
    # Converts literal string "nan" or empty strings into actual pandas NA values
    df_csv['domain'] = df_csv['domain'].replace(['', '-'], pd.NA)
    df_json['domain'] = df_json['domain'].replace(['', '-'], pd.NA)


    # 4. Standardize the strict match columns to strings
    strict_columns = ['dst', 'dstport', 'srcport']
    for col in strict_columns:
        df_csv[col] = df_csv[col].astype(str)
        df_json[col] = df_json[col].astype(str)

    print(f"[*] Correlating data with a {time_delta_seconds}s time window...")
    
    # 5. STEP ONE: The Strict Merge (Ignoring Domain for a moment)
    # Keeps every row from df_json and finds the nearest match in df_csv based on timestamp and strict columns, where it adds application and domain info from df_csv if a match is found within the time window.
    merged = pd.merge_asof(
        df_json,
        df_csv,
        on='timestamp',
        by=['dst', 'dstport', 'srcport'],
        direction='nearest',
        tolerance=pd.Timedelta(seconds=time_delta_seconds),
        suffixes=('_json', '_csv')
    )


    # 6. STEP TWO: Best Effort Domain Validation
    # We find rows where BOTH domains exist, but they DO NOT match.
    if domain_aware:
        domain_conflict = (
            merged['domain_json'].notna() & 
            merged['domain_csv'].notna() & 
            (merged['domain_json'] != merged['domain_csv'])
        )

        # If there is a conflict, we strip the 'app' match because it's a false positive
        conflicts = domain_conflict.sum()
        if conflicts > 0:
            logger.info("Removed %d potential matches due to conflicting domain names.", conflicts)
            conflict_rows = merged.loc[
                domain_conflict,
                ['timestamp', 'srcport', 'dst', 'dstport', 'domain_json', 'domain_csv']
            ]
            for _, row in conflict_rows.iterrows():
                logger.info(
                    "Merge conflict at timestamp=%s srcport=%s dst=%s dstport=%s | domain in json: %s | domain in csv: %s",
                    row['timestamp'],
                    row['srcport'],
                    row['dst'],
                    row['dstport'],
                    row['domain_json'],
                    row['domain_csv']
                )

    
        merged.loc[domain_conflict, 'app'] = None

    # 7. Format the output to your exact JSON schema
    final_df = pd.DataFrame({
        "application": merged['app'],
        "ja4_fingerprint": merged['JA4.1'],
        "ja4_fingerprint_string": merged['JA4_r.1'],
        "ja4s_fingerprint": merged['JA4S']
    })

    # Replace Pandas NaNs with None for proper JSON 'null' output
    final_df = final_df.astype(object).where(pd.notna(final_df), None)

    # 8. Export
    output_records = final_df.to_dict(orient='records')

    # Safety pass: ensure any remaining float NaN is converted to None
    for record in output_records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None

    with open(output_file, 'w') as f:
        json.dump(output_records, f, indent=4, allow_nan=False)
        
    print(f"[+] Success! Exported {len(output_records)} records to {output_file}")

if __name__ == "__main__":
    correlate_traffic_best_effort('sysmon_data.csv', 'network_data.json', 'correlated_ja4_db.json', time_delta_seconds=2)