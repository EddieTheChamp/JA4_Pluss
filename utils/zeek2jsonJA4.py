"""
Bruk: py zeek2jsonJA4.py -a <application_name> -ssl <ssl_log_path> [-conn <conn_log_path>]
Eksempel: py zeek2jsonJA4.py -a "Outlook" -ssl ssl.log -conn conn.log > output.json
"""

import sys
import argparse
import json


separator = "\t"  # Zeek bruker tabulator som separator i loggene

def clean_value(val):
    # Zeek bruker "-" for tomme felt.
    if val is None or val == "-" or val == "(empty)":
        return None
    return val

def get_fields_from_log(log_path):
    with open(log_path, "r") as f:
        for line in f:
            if line.startswith("#fields"):
                return line.strip().split(separator)[1:]
    return None

def main():
    parser = argparse.ArgumentParser(description="Extracts JA4, JA4s, JA4t, JA4ts from Zeek SSL and Conn logs to JSON stdout.")
    parser.add_argument("-a", "--application_name", required=True)
    parser.add_argument("-ssl", "--ssl_log", required=True)
    parser.add_argument("-conn", "--conn_log")
    args = parser.parse_args()

    # 1. Forbered Conn-data (hvis filen finnes)
    # Vi lager en ordbok: { "UID": {"ja4t": "...", "ja4ts": "..."} }
    conn_map = {}
    if args.conn_log:
        conn_fields = get_fields_from_log(args.conn_log)
        if conn_fields:
            with open(args.conn_log, "r") as f:
                for line in f:
                    if line.startswith("#"): continue
                    parts = line.strip().split(separator)
                    if len(parts) == len(conn_fields):
                        c_entry = dict(zip(conn_fields, parts))
                        uid = c_entry.get("uid")
                        if uid:
                            conn_map[uid] = {
                                "ja4t": c_entry.get("ja4t"),
                                "ja4ts": c_entry.get("ja4ts")
                            }

    # 2. Prosesser SSL-loggen linje for linje
    ssl_fields = get_fields_from_log(args.ssl_log)
    if not ssl_fields:
        print("Error: could not find #fields in SSL log", file=sys.stderr)
        return
    
    print("[")  # Start JSON-arrayen
    first_entry = True
    with open(args.ssl_log, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            
            values = line.strip().split(separator)
            if len(values) != len(ssl_fields):
                continue

            ssl_entry = dict(zip(ssl_fields, values))
            uid = ssl_entry.get("uid")

            # 3. Hent "bonus-data" fra conn_map hvis UID finnes der
            extra_data = conn_map.get(uid, {})
            
            # 4. Bygg og print resultatet med en gang
            output_data = {
                "application": args.application_name,
                "ja4": clean_value(ssl_entry.get("ja4")),
                "ja4s": clean_value(ssl_entry.get("ja4s")),
                "ja4t": clean_value(extra_data.get("ja4t")),
                "ja4ts": clean_value(extra_data.get("ja4ts")),
                "sni": clean_value(ssl_entry.get("server_name")),
            }
            
            # Vi printer bare hvis vi i det minste har en JA4
            if output_data["ja4"]:
                if not first_entry:
                    print(",")  # Legg til komma mellom objektene
                print(json.dumps(output_data), end="")
                first_entry = False
    print("\n]")  # Avslutt JSON-arrayen
if __name__ == "__main__":
    main()