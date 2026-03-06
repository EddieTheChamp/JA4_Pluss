# Project Dictionary

This workspace evaluates exactly how well we can identify network applications (like Firefox, Chrome, Emotet) purely from their encrypted TLS fingerprints using JA4 methodology.

We test four distinct approaches, each built iteratively on top of the last.

## Repository Structure

### 🗃️ `Create Dictionary/`
The raw dataset ingress folder. All traffic parsing starts here. The current live dataset is `correlated_ja4_db_large.json`.

### 🔄 `Iteration1/`
**Baseline Evaluation.**
This takes the official, public FoxIO JA4+ database (`ja4+_db.json`) and queries it using our test traffic to see what its baseline accuracy is.
*   `iteration1.py`: Extracts exact match accuracies for `ja4`, `ja4_ja4s`, and `ja4_ja4s_ja4ts` strictness modes. 

### 📚 `Iteration2/`
**Custom Dictionary & Prototype Estimator.**
Instead of using the FoxIO Dictionary, we build our *own* dictionary database composed exclusively of the 80% training split of our specific traffic. We then compare our custom DB against the public FoxIO DB.
*   `iteration2.py`: Silently spins up the custom dataset, tests it, and generates side-by-side Top-K and Collision comparative graphs in the `Visualization/` folder.
*   `prototype_predictor.py`: A standalone, live-streaming prototype tool. Feed it any blind JSON traffic stream, and it will print live terminal predictions using the custom database.

### 🤖 `Iteration3/`
**Machine Learning: Random Forest (Application Prediction).**
Moving away from exact-match dictionaries entirely. We parse the 36-character JA4 string into 8 modular features (TLS version, ciphers count, ALPN, etc.) and train a Random Forest model to predict the application. 
*   `iteration3_ml.py`: Trains the RF model, evaluates the test split, and dynamically adds its results as a third pillar to the collision and Top-K graphs generated in Iteration 2.

### 🦠 `Iteration4/`
**Machine Learning: Bot & Malware Modularity.**
This iteration focuses exclusively on using the modular breakdown of JA4 (`a_b_c`) to classify if an unknown traffic fingerprint belongs to a Bot or a Human, rather than trying to guess the exact application name.
