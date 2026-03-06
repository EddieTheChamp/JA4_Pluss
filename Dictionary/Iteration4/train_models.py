import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from data_loader import load_and_preprocess_data, prepare_datasets

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    # Train stats
    y_pred_tr = model.predict(X_train)
    acc_tr = accuracy_score(y_train, y_pred_tr)
    prec_tr = precision_score(y_train, y_pred_tr, zero_division=0)
    rec_tr = recall_score(y_train, y_pred_tr, zero_division=0)
    f1_tr = f1_score(y_train, y_pred_tr, zero_division=0)
    
    # Test stats
    y_pred_te = model.predict(X_test)
    acc_te = accuracy_score(y_test, y_pred_te)
    prec_te = precision_score(y_test, y_pred_te, zero_division=0)
    rec_te = recall_score(y_test, y_pred_te, zero_division=0)
    f1_te = f1_score(y_test, y_pred_te, zero_division=0)
    
    print(f"--- {name} ---")
    print(f"TRAIN: Accuracy: {acc_tr:.4f}  Precision: {prec_tr:.4f}  Recall: {rec_tr:.4f}  F1 Score: {f1_tr:.4f}")
    print(f"TEST:  Accuracy: {acc_te:.4f}  Precision: {prec_te:.4f}  Recall: {rec_te:.4f}  F1 Score: {f1_te:.4f}\n")
    return model

def train_and_evaluate(task_name, X_train, X_test, y_train, y_test):
    print(f"\n========== Training for Task: {task_name.upper()} ==========")
    if sum(y_train) == 0:
        print(f"Warning: No positive samples in training data for {task_name}. Skipping training as there is only one class.")
        return {}
        
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=100, verbose=0, random_state=42, auto_class_weights='Balanced')
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    return trained_models

if __name__ == "__main__":
    df = load_and_preprocess_data("../Dictionary/ja4+_db.json")
    if len(df) == 0:
        print("No valid data loaded.")
        exit(0)
        
    datasets = prepare_datasets(df)
    
    X_train, X_test, y_train, y_test = datasets["bot"]
    models_bot = train_and_evaluate("Bot vs Benign", X_train, X_test, y_train, y_test)
    
    all_importances = {"bot": {}}
    if models_bot:
        for model_name, model in models_bot.items():
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_.tolist()
                all_importances["bot"][model_name] = imp
                
    output_data = {
        "features": list(datasets["feature_names"]),
        "importances": all_importances
    }
    with open("feature_importances.json", "w") as f:
        json.dump(output_data, f)
    print("\nSaved feature_importances.json")
