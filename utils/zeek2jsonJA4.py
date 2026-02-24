"""
Bruk: py zeek2jsonJA4.py -a <application_name> -ssl <ssl_log_path> [-conn <conn_log_path>]
Eksempel: py zeek2jsonJA4.py -a "Outlook" -ssl ssl.log -conn conn.log
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
                return line.strip().split()[1:]
    return None

def main():
    parser = argparse.ArgumentParser(description="Kobler Zeek SSL og Conn logger for JA4 analyse")
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
                    parts = line.strip().split()
                    if len(parts) == len(conn_fields):
                        c_entry = dict(zip(conn_fields, parts))
                        uid = c_entry.get("uid")
                        if uid:
                            # Vi lagrer bare det vi trenger for å spare minne
                            conn_map[uid] = {
                                "ja4t": c_entry.get("ja4t"),
                                "ja4ts": c_entry.get("ja4ts")
                            }

    # 2. Prosesser SSL-loggen linje for linje
    ssl_fields = get_fields_from_log(args.ssl_log)
    if not ssl_fields:
        print("Feil: Fant ikke #fields i SSL-loggen", file=sys.stderr)
        return

    with open(args.ssl_log, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            
            values = line.strip().split()
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
                "sni": clean_value(ssl_entry.get("server_name"))
            }
            
            # Vi printer bare hvis vi i det minste har en JA4
            if output_data["ja4"]:
                print(json.dumps(output_data))

if __name__ == "__main__":
    main()