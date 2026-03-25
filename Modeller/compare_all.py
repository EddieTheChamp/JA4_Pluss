"""Compare all three models and generate visualizations.

Runs eval_foxio, eval_egenlagd, and eval_random_forest on the same test split
then generates:
  - confusion_matrix_FoxIO.png
  - confusion_matrix_Egenlagd.png
  - confusion_matrix_RandomForest.png
  - comparative_top_k_accuracy.png
  - comparative_collision_matrix.png
  - comparison_results.json   (summary table as JSON)

All output goes to the results/ folder.

Usage:
    python compare_all.py
"""

from __future__ import annotations
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay
    _PLOT_OK = True
except ImportError:
    _PLOT_OK = False

import eval_foxio
import eval_egenlagd
import eval_random_forest
import eval_pipeline

_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = _ROOT / "results"


# ── helpers ───────────────────────────────────────────────────────────────────

def _top_k_hit(true_app: str, top_k: list[str], k: int) -> bool:
    return true_app.lower() in [t.lower() for t in top_k[:k]]


def _compute_metrics(results: list[dict], model_name: str) -> dict:
    total = len(results)
    if total == 0:
        return {}
    top1  = sum(1 for r in results if _top_k_hit(r["true_app"], r["top_k"], 1))
    top3  = sum(1 for r in results if _top_k_hit(r["true_app"], r["top_k"], 3))
    top5  = sum(1 for r in results if _top_k_hit(r["true_app"], r["top_k"], 5))
    unique    = sum(1 for r in results if r["matches_count"] == 1)
    collision = sum(1 for r in results if r["matches_count"] > 1)
    unknown   = sum(1 for r in results if r["matches_count"] == 0)
    return {
        "model": model_name,
        "total": total,
        "top1_pct":      round(top1  / total * 100, 1),
        "top3_pct":      round(top3  / total * 100, 1),
        "top5_pct":      round(top5  / total * 100, 1),
        "unique_pct":    round(unique    / total * 100, 1),
        "collision_pct": round(collision / total * 100, 1),
        "unknown_pct":   round(unknown   / total * 100, 1),
    }


# ── confusion matrix ─────────────────────────────────────────────────────────

def _plot_confusion_matrix(results: list[dict], model_name: str, top_n: int = 10) -> None:
    if not _PLOT_OK:
        return
    class_counts: dict[str, int] = {}
    for r in results:
        class_counts[r["true_app"]] = class_counts.get(r["true_app"], 0) + 1
    top_classes = [c for c, _ in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    if "Unknown" not in top_classes:
        top_classes.append("Unknown")

    y_true, y_pred = [], []
    for r in results:
        t = r["true_app"] if r["true_app"] in top_classes else "Other"
        p = r["prediction"] if r["prediction"] in top_classes else "Other"
        y_true.append(t)
        y_pred.append(p)

    labels = sorted(set(y_true + y_pred))
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, display_labels=labels,
        ax=ax, cmap="Blues", xticks_rotation="vertical",
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved → {out.name}")


# ── top-k bar chart ─────────────────────────────────────────────────────────

def _plot_top_k(all_metrics: list[dict]) -> None:
    if not _PLOT_OK:
        return
    labels = ["Top-1", "Top-3", "Top-5"]
    x = np.arange(len(labels))
    width = 0.20
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(all_metrics):
        bars = ax.bar(x + (i - 1.5) * width,
                      [m["top1_pct"], m["top3_pct"], m["top5_pct"]],
                      width, label=m["model"], color=colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-K Accuracy: FoxIO vs Egenlagd vs Random Forest", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 115)
    plt.tight_layout()
    out = RESULTS_DIR / "comparative_top_k_accuracy.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved → {out.name}")


# ── collision stacked bar chart ──────────────────────────────────────────────

def _plot_collision(all_metrics: list[dict]) -> None:
    if not _PLOT_OK:
        return
    names = [m["model"] for m in all_metrics]
    unique    = np.array([m["unique_pct"]    for m in all_metrics])
    collision = np.array([m["collision_pct"] for m in all_metrics])
    unknown   = np.array([m["unknown_pct"]   for m in all_metrics])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(names, unique,    label="Unique match",      color="#2ca02c")
    ax.bar(names, collision, label="Collision (>1 app)", color="#ff7f0e", bottom=unique)
    ax.bar(names, unknown,   label="Unknown / Ambiguous",color="#d62728", bottom=unique + collision)

    # Annotate segments
    for i, (u, col) in enumerate(zip(unique, collision)):
        if u > 3:
            ax.text(i, u / 2, f"{u:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        if col > 3:
            ax.text(i, u + col / 2, f"{col:.1f}%", ha="center", va="center", color="white", fontweight="bold")

    ax.set_ylabel("Percentage of test queries (%)")
    ax.set_title("Match Disambiguation: FoxIO vs Egenlagd vs Random Forest", fontsize=13)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = RESULTS_DIR / "comparative_collision_matrix.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved → {out.name}")


# ── summary table ─────────────────────────────────────────────────────────────

def _print_table(all_metrics: list[dict]) -> None:
    header = f"{'Model':<24} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'Unique':>9} {'Collision':>11} {'Unknown':>9}"
    print("\n" + "=" * len(header))
    print("  JA4+ Model Comparison".center(len(header)))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(
            f"{m['model']:<24} "
            f"{m['top1_pct']:>7.1f}% "
            f"{m['top3_pct']:>7.1f}% "
            f"{m['top5_pct']:>7.1f}% "
            f"{m['unique_pct']:>8.1f}% "
            f"{m['collision_pct']:>10.1f}% "
            f"{m['unknown_pct']:>8.1f}%"
        )
    print("=" * len(header) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    if not _PLOT_OK:
        print("[compare_all] WARNING: matplotlib/sklearn not installed — charts will be skipped.")

    print("\n[compare_all] Running FoxIO evaluation...")
    foxio_res = eval_foxio.run()

    print("\n[compare_all] Running Egenlagd evaluation...")
    egenlagd_res = eval_egenlagd.run()

    print("\n[compare_all] Running Random Forest evaluation...")
    rf_res = eval_random_forest.run()

    print("\n[compare_all] Running Unified Pipeline evaluation...")
    pipeline_res_raw = eval_pipeline.run()
    # Map from eval_pipeline format back to compare_all format
    pipeline_res = []
    for r in pipeline_res_raw:
        # Pipeline doesn't have matches_count in the same way, but it has source
        mc = 1 if r["source"] in ("egenlagd_exact", "ambiguous_resolved_by_rf", "random_forest") else 0
        pipeline_res.append({
            "true_app": r["true_app"],
            "prediction": r["pred_app"],
            "top_k": r["top_k"], # use the full candidate list
            "matches_count": mc,
        })

    all_results = [
        ("FoxIO",         foxio_res),
        ("Egenlagd",      egenlagd_res),
        ("Random Forest", rf_res),
        ("Unified Pipeline", pipeline_res),
    ]

    print("\n[compare_all] Generating charts...")
    all_metrics = []
    for name, results in all_results:
        _plot_confusion_matrix(results, name)
        all_metrics.append(_compute_metrics(results, name))

    _plot_top_k(all_metrics)
    _plot_collision(all_metrics)
    _print_table(all_metrics)

    out = RESULTS_DIR / "comparison_results.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[compare_all] Summary saved → {out}")


if __name__ == "__main__":
    run()
