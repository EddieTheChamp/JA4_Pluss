import argparse
import subprocess
import sys
import os

def run_step(command, step_name):
    print(f"\n{'='*50}")
    print(f"Executing: {step_name}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*50}")
    
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    else:
        print(f"\n[SUCCESS] '{step_name}' completed.")

def main():
    parser = argparse.ArgumentParser(description="Run the full JA4 Evaluation Pipeline.")
    parser.add_argument("--dataset_file", default="raw_test_samples.json", help="Path to the main JSON dataset file")
    parser.add_argument("--foxio_db", default="ja4+_db.json", help="Path to the official FoxIO DB")
    args = parser.parse_args()
    
    dataset_file = args.dataset_file
    foxio_db = args.foxio_db
    
    if not os.path.exists(dataset_file):
        print(f"[ERROR] Required dataset file not found: {dataset_file}")
        sys.exit(1)

    print("\nStarting JA4 Application Identification Evaluation Pipeline...")

    # =========================================================================
    # STEP 1: Build Custom Dictionary Database (Egenlagd)
    # 
    # Purpose: Create our own dictionary database using strictly the training
    #          split (80%) of the provided dataset. We remove duplicates so
    #          that it is structured just like the official FoxIO DB but tailored
    #          to our own captured traffic labels.
    # =========================================================================
    run_step([
        "python", "build_custom_db.py",
        "--dataset_file", dataset_file,
        "--output_db", "egenlagd_db.json"
    ], "Build Custom Database")

    # =========================================================================
    # STEP 2: Evaluate Official FoxIO Dictionary
    #
    # Purpose: We run our unified `dictionary_model.py` passing in the official
    #          database (`ja4+_db.json`). It will run an 80/20 train/test split
    #          on `dataset_file`, ignore the training data, and strictly evaluate
    #          the unseen 20% test samples against the FoxIO dictionary. The
    #          results are saved to a specific payload JSON for graph generation.
    # =========================================================================
    run_step([
        "python", "dictionary_model.py",
        "--dataset_file", dataset_file,
        "--db_file", foxio_db,
        "--model_name", "FoxIO",
        "--output_file", "Results/dictionary_FoxIO_result.json"
    ], "Evaluate FoxIO (Official JA4 Database)")
    
    # =========================================================================
    # STEP 3: Evaluate Custom Dictionary (Egenlagd)
    #
    # Purpose: Exactly like Step 2, but instead of using the official FoxIO DB,
    #          we pass in the `egenlagd_db.json` we just created in Step 1.
    #          This allows us to see how well our own exact-matching logic works.
    # =========================================================================
    run_step([
        "python", "dictionary_model.py",
        "--dataset_file", dataset_file,
        "--db_file", "egenlagd_db.json",
        "--model_name", "Egenlagd",
        "--output_file", "Results/dictionary_Egenlagd_result.json"
    ], "Evaluate Egenlagd (Custom Trained Database)")

    # =========================================================================
    # STEP 4: Evaluate Random Forest Machine Learning Model
    #
    # Purpose: Instead of an exact-match dictionary, this trains a Scikit-Learn
    #          Random Forest classifier on the same 80% training data, and then
    #          predicts the remaining 20% test data. It calculates confidence
    #          probabilities to determine the Top-K matches.
    # =========================================================================
    run_step([
        "python", "random_forest_model.py",
        "--dataset_file", dataset_file,
        "--output_file", "Results/Random_Forest_result.json"
    ], "Evaluate Random Forest Machine Learning Model")

    # =========================================================================
    # STEP 5: Generate Comparison Visualizations
    #
    # Purpose: Reads the three output JSON payload files from Steps 2, 3, and 4
    #          (FoxIO, Egenlagd, and Random Forest). It calculates classification
    #          metrics (accuracy, precision, recall) and uses Matplotlib to draw
    #          Confusion Matrices, Top-K accuracy charts, and Collision matrices.
    #          These are saved as PNG files in the working directory.
    # =========================================================================
    run_step([
        "python", "Visualization/generate_comparison_graphs.py",
        "--foxio_file", "Results/dictionary_FoxIO_result.json",
        "--egenlagd_file", "Results/dictionary_Egenlagd_result.json",
        "--rf_file", "Results/Random_Forest_result.json"
    ], "Generate Comparison Graphs")

    print(f"\n{'='*50}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("All results payload JSONs and graphical outputs have been updated.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
