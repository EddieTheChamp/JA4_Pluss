import tkinter as tk
from tkinter import ttk, messagebox
import json
import traceback
from pathlib import Path
from collections import defaultdict
import re

_ROOT = Path(__file__).resolve().parent

def get_foxio_db_path() -> Path:
    # Try a few common locations where the ja4+_db.json might be located
    candidates = [
        _ROOT / "data" / "models" / "ja4+_db.json",
        _ROOT.parent / "Dictionary" / "ja4+_db.json",
        _ROOT.parent / "ja4+_db.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # Return default if not found (to show error in load_db)

FOXIO_DB_PATH = get_foxio_db_path()

class FoxIOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FoxIO Database Lookup")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#2c3e50")
        
        self.db = []
        self.indexes = {
            "ja4": defaultdict(list),
            "ja4s": defaultdict(list),
            "ja4t": defaultdict(list),
            "ja4ts": defaultdict(list),
            "ja4h": defaultdict(list),
            "ja4x": defaultdict(list)
        }
        
        self.create_widgets()
        
        self.status_var.set("Loading database...")
        self.root.after(100, self.load_db) # Load db asynchronously after UI renders

    def load_db(self):
        try:
            if not FOXIO_DB_PATH.exists():
                messagebox.showerror("Error", f"FoxIO DB not found at {FOXIO_DB_PATH}\nPlease ensure the database file is placed correctly.")
                self.status_var.set("Database load failed.")
                return
            with open(FOXIO_DB_PATH, "r", encoding="utf-8") as f:
                self.db = json.load(f)
            
            for entry in self.db:
                for k in self.indexes.keys():
                    val = entry.get(k) or entry.get(f"{k}_fingerprint")
                    if val:
                        self.indexes[k][val.strip().lower()].append(entry)
            self.status_var.set(f"Database loaded: {len(self.db)} entries.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load DB:\n{traceback.format_exc()}")
            self.status_var.set("Database load failed.")

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="FoxIO Database Individual Lookup", style="Title.TLabel").pack(pady=(0, 20))
        
        input_frame = ttk.LabelFrame(main_frame, text="Input Fingerprints (Paste JSON Snippet)", padding="15 15 15 15")
        input_frame.pack(fill=tk.X, pady=(0, 20))
        input_frame.columnconfigure(0, weight=1)
        
        self.txt_input = tk.Text(input_frame, height=8, width=80, font=("Consolas", 10), bg="#fdfdfd")
        self.txt_input.grid(row=0, column=0, sticky=tk.EW, pady=5)
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=1, column=0, pady=10)
        ttk.Button(btn_frame, text="Lookup in FoxIO", command=self.on_lookup, width=20).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Clear", command=self.on_clear, width=10).pack(side=tk.LEFT, padx=10)
        
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(main_frame, textvariable=self.status_var, foreground="#7f8c8d").pack(anchor=tk.W, pady=(0, 10))

        self.result_frame = ttk.LabelFrame(main_frame, text="Lookup Results", padding="15 15 15 15")
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("application", "os", "device", "notes")
        self.tree = ttk.Treeview(self.result_frame, columns=columns, show="tree headings")
        self.tree.heading("#0", text="Type / Fingerprint")
        self.tree.heading("application", text="Application")
        self.tree.heading("os", text="OS")
        self.tree.heading("device", text="Device")
        self.tree.heading("notes", text="Notes")
        
        self.tree.column("#0", width=350)
        self.tree.column("application", width=200)
        self.tree.column("os", width=120)
        self.tree.column("device", width=120)
        self.tree.column("notes", width=150)
        
        scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def on_clear(self):
        self.txt_input.delete(1.0, tk.END)
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.status_var.set("Cleared.")

    def on_lookup(self):
        if not self.db:
            messagebox.showwarning("Wait", "Database is not loaded yet or failed to load.")
            return

        raw_text = self.txt_input.get(1.0, tk.END).strip()
        if not raw_text:
            messagebox.showwarning("Input Error", "Please paste a JSON snippet to lookup.")
            return

        parsed = {}
        # Try brute-force regex first since it handles partial/broken JSON robustly
        for match in re.finditer(r'"([^"]+)"\s*:\s*"([^"]+)"', raw_text):
            parsed[match.group(1)] = match.group(2)

        def _get(keys):
            for k in keys:
                if k in parsed:
                    return parsed[k]
            return None

        queries = {
            "ja4": _get(["ja4", "ja4_fingerprint"]),
            "ja4s": _get(["ja4s", "ja4s_fingerprint"]),
            "ja4t": _get(["ja4t", "ja4t_fingerprint"]),
            "ja4ts": _get(["ja4ts", "ja4ts_fingerprint"]),
            "ja4h": _get(["ja4h", "ja4h_fingerprint"]),
            "ja4x": _get(["ja4x", "ja4x_fingerprint"])
        }

        # Clear existing tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        found_any = False
        total_hits_shown = 0

        for ftype, val in queries.items():
            if not val:
                continue
            
            found_any = True
            hits = self.indexes[ftype].get(val.strip().lower(), [])
            
            if not hits:
                # Insert a "not found" node
                self.tree.insert("", tk.END, text=f"{ftype.upper()}: {val}", values=("Not Found", "-", "-", "-"), tags=("not_found",))
                continue
                
            # Deduplicate hits by application
            unique_apps = {}
            for h in hits:
                app = h.get("application") or h.get("Application") or "Unknown"
                if app not in unique_apps:
                    unique_apps[app] = h
            
            # Root node for this FP
            node = self.tree.insert("", tk.END, text=f"{ftype.upper()}: {val}", values=(f"{len(unique_apps)} unique app(s)", "-", "-", "-"), tags=("header",))
            
            # Insert deduplicated children
            for app, h in unique_apps.items():
                os_val = h.get("os") or h.get("OS") or "-"
                dev_val = h.get("device") or h.get("Device") or "-"
                notes_val = h.get("notes") or h.get("Notes") or "-"
                self.tree.insert(node, tk.END, text="  ↳ Hit", values=(app, os_val, dev_val, notes_val))
                total_hits_shown += 1
                
            # Open the node
            self.tree.item(node, open=True)

        if not found_any:
            messagebox.showwarning("Input Error", "Could not find any valid JA4 fields in the input text. Ensure it is formatted like '\"ja4_fingerprint\": \"...\"'.")
            return
            
        self.tree.tag_configure("header", font=("Segoe UI", 10, "bold"), background="#e0e0e0")
        self.tree.tag_configure("not_found", foreground="red")
        
        self.status_var.set(f"Lookup completed. Displaying {total_hits_shown} unique results.")

def main():
    root = tk.Tk()
    app = FoxIOGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
