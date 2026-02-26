import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_confusion_matrix(y_true, y_pred, model_name, top_n_classes=10):
    # Determine top classes
    class_counts = {}
    for label in y_true:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    top_classes = [c[0] for c in sorted_classes[:top_n_classes]]
    
    if "Unknown" not in top_classes and "Unknown" in y_pred:
        top_classes.append("Unknown")

    y_true_filtered = []
    y_pred_filtered = []
    for t, p in zip(y_true, y_pred):
        if t in top_classes and p in top_classes:
            y_true_filtered.append(t)
            y_pred_filtered.append(p)
        else:
            y_true_filtered.append(t if t in top_classes else "Other")
            y_pred_filtered.append(p if p in top_classes else "Other")

    display_labels = sorted(list(set(y_true_filtered + y_pred_filtered)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true_filtered, 
        y_pred_filtered, 
        labels=display_labels,
        display_labels=display_labels, 
        ax=ax, 
        cmap="Blues",
        xticks_rotation='vertical'
    )
    
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    
    filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    filepath = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"[{model_name}] Saved Confusion Matrix to {filepath}")

def plot_top_k(foxio_res, egenlagd_res, rf_res):
    models = ["foxio", "egenlagd", "random_forest"]
    labels = ["Top-1 Match", "Top-3 Match", "Top-5 Match"]
    
    scores = {}
    for model_name, results_list in zip(models, [foxio_res, egenlagd_res, rf_res]):
        top1 = 0
        top3 = 0
        top5 = 0
        total = len(results_list)
        for r in results_list:
            true_app = r["true_app"]
            top_k_list = r.get("top_k", [])
            
            if true_app in top_k_list[:1]: top1 += 1
            if true_app in top_k_list[:3]: top3 += 1
            if true_app in top_k_list[:5]: top5 += 1
            
        scores[model_name] = [
            (top1/total*100) if total > 0 else 0, 
            (top3/total*100) if total > 0 else 0, 
            (top5/total*100) if total > 0 else 0
        ]
        
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, scores["foxio"], width, label='Dictionary (FoxIO)', color='#1f77b4')
    rects2 = ax.bar(x, scores["egenlagd"], width, label='Dictionary (Egenlagd)', color='#ff7f0e')
    rects3 = ax.bar(x + width, scores["random_forest"], width, label='Random Forest', color='#2ca02c')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-K Accuracy (Dictionaries vs Random Forest)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 110)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    filepath = os.path.join(SCRIPT_DIR, 'comparative_top_k_accuracy.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved Top-K comparative chart to {filepath}")

def plot_collision_matrix(foxio_res, egenlagd_res, rf_res):
    models = ["foxio", "egenlagd", "random_forest"]
    display_names = ["Dictionary (FoxIO)", "Dictionary (Egenlagd)", "Random Forest"]
    
    unique_matches = []
    collision_matches = []
    unknowns = []

    for model_name, results_list in zip(models, [foxio_res, egenlagd_res, rf_res]):
        unique = 0
        collision = 0
        unknown = 0
        for r in results_list:
            matches = r.get("matches_count", 0)
            if matches == 1:
                unique += 1
            elif matches > 1:
                collision += 1
            else:
                unknown += 1
        
        total = unique + collision + unknown
        if total > 0:
            unique_matches.append(unique/total*100)
            collision_matches.append(collision/total*100)
            unknowns.append(unknown/total*100)
        else:
            unique_matches.append(0)
            collision_matches.append(0)
            unknowns.append(0)
        
    unique_matches = np.array(unique_matches)
    collision_matches = np.array(collision_matches)
    unknowns = np.array(unknowns)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(display_names, unique_matches, label='Unique Match (Only 1 App)', color='#2ca02c')
    ax.bar(display_names, collision_matches, bottom=unique_matches, label='Collision (>1 Apps Match)', color='#ff7f0e')
    ax.bar(display_names, unknowns, bottom=unique_matches + collision_matches, label='Unknown (Not in DB)', color='#d62728')

    ax.set_ylabel('Percentage of Queries (%)')
    ax.set_title('Match Disambiguation: Dictionaries vs Random Forest')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for i in range(len(models)):
        if unique_matches[i] > 5:
            ax.text(i, unique_matches[i]/2, f'{unique_matches[i]:.1f}%', ha='center', va='center', color='white', fontweight='bold')
        if collision_matches[i] > 5:
            ax.text(i, unique_matches[i] + collision_matches[i]/2, f'{collision_matches[i]:.1f}%', ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(SCRIPT_DIR, 'comparative_collision_matrix.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved Comparative Collision Matrix to {filepath}")

def load_result_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find {filepath}. Passing empty list.")
        return []

def run_reports(foxio_file, egenlagd_file, rf_file):
    print("Loading component files...")
    foxio_res = load_result_file(foxio_file)
    egenlagd_res = load_result_file(egenlagd_file)
    rf_res = load_result_file(rf_file)

    # Use FoxIO file (or whichever is populated) as ground truth map if needed
    baseline_res = foxio_res if foxio_res else (egenlagd_res if egenlagd_res else rf_res)
    if not baseline_res:
        print("ERROR: All three result files were empty or missing. Cannot generate graphs.")
        return

    print(f"Loaded {len(baseline_res)} evaluated samples.")
    
    # Generate Confusion Matrices for all 3
    if foxio_res:
        y_true_foxio = [r["true_app"] for r in foxio_res]
        y_pred_foxio = [r["prediction"] for r in foxio_res]
        plot_confusion_matrix(y_true_foxio, y_pred_foxio, "Dictionary_FoxIO")
    
    if egenlagd_res:
        y_true_egenlagd = [r["true_app"] for r in egenlagd_res]
        y_pred_egenlagd = [r["prediction"] for r in egenlagd_res]
        plot_confusion_matrix(y_true_egenlagd, y_pred_egenlagd, "Dictionary_Egenlagd")
    
    if rf_res:
        y_true_rf = [r["true_app"] for r in rf_res]
        y_pred_rf = [r["prediction"] for r in rf_res]
        plot_confusion_matrix(y_true_rf, y_pred_rf, "Random_Forest")
    
    # Generate aggregate metrics
    plot_top_k(foxio_res, egenlagd_res, rf_res)
    plot_collision_matrix(foxio_res, egenlagd_res, rf_res)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3-Model JA4 Visualizations.")
    parser.add_argument("--foxio_file", default=os.path.join(SCRIPT_DIR, "..", "Results", "dictionary_FoxIO_result.json"), help="Path to the FoxIO results")
    parser.add_argument("--egenlagd_file", default=os.path.join(SCRIPT_DIR, "..", "Results", "dictionary_Egenlagd_result.json"), help="Path to the Egenlagd results")
    parser.add_argument("--rf_file", default=os.path.join(SCRIPT_DIR, "..", "Results", "Random_Forest_result.json"), help="Path to the Random Forest results")
    
    args = parser.parse_args()
    run_reports(args.foxio_file, args.egenlagd_file, args.rf_file)
