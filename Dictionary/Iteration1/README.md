# Iteration 1

This directory contains `iteration1.py`, a script used to analyze a raw JSON array of JA4 traffic fingerprints against the JA4+ database and summarize the matches.

## How to Run

By default, the script expects the JA4+ database file `ja4+_db.json` to be located in the parent directory (`../ja4+_db.json`).

### Running from the Iteration1 directory


```bash
python iteration1.py --capture_file firefox.json
```

### Arguments
You can override the default file paths using these arguments:
* `--capture_file`: Path to the raw capture JSON file (default: `iteration1_capture.json`).
* `--db_file`: Path to the JA4+ database JSON file (default: `../ja4+_db.json`).
