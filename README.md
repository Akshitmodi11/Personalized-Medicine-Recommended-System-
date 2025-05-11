# Personalized Medication Recommendation System

A full-stack intelligent healthcare application that predicts personalized medication codes based on a patient's demographics, conditions, procedures, allergies, and other clinical metadata using a trained machine learning model. The application integrates a user-friendly front end with a CatBoost-based multi-label classifier backend powered by Flask.

---

## Project Overview

Traditional clinical decision support systems often lack personalization when suggesting medications. This project addresses that gap by building a **personalized medication recommendation system**, trained on structured synthetic healthcare data.

The project pipeline includes:

- EDA and preprocessing in Jupyter Notebooks
- Multi-hot and categorical encoding for feature preparation
- Dimensionality reduction via Truncated SVD
- Training a Classifier Chain model with a CatBoost base estimator
- Real-time inference through a Flask web app
- Enhanced UX with autocomplete via BioPortal and RxNorm integration

## 📺Demo Video

[![Watch the Demo](https://img.youtube.com/vi/q80QigVE26o/hqdefault.jpg)](https://youtu.be/q80QigVE26o)

---

## 📂 Project Structure

```bash
PMRS/
├── app.py                        # Flask backend application
├── index.html                   # Main UI frontend form
├── EDA.ipynb                    # Exploratory Data Analysis
├── FINAL_MODEL_TRAINING.ipynb   # Full pipeline: preprocessing → model training
├── code_adding.ipynb            # Code snippets for integrations
├── static/
│   ├── style.css                # Custom CSS
│   ├── condition_map.json       # Map of condition descriptions to SNOMED codes
│   └── allergy_map.json         # Allergy description-code map

├── final model/
│   ├── medication_chain_catboost.pkl
│   ├── mlb_medication.pkl
│   ├── mlb_procedure.pkl
│   ├── mlb_condition.pkl
│   ├── mlb_allergy.pkl
│   ├── label_encoders.pkl
│   ├── svd_proc.pkl
│   ├── svd_cond.pkl
│   └── svd_alg.pkl
└── patient_records.csv
        # Logged patient entries and predictions
```

---

## Data Pipeline

### 1. **Preprocessing:**
- Patient records include fields like age, gender, city, race, ethnicity, height, weight, marital status, smoking status, etc.
- Multi-label columns: `PROCEDURE_CODES`, `CONDITION_CODES`, `ALLERGIES_CODE`, and `MEDICATION_CODES`.
- Encoded using:
  - `MultiLabelBinarizer` for multi-hot inputs
  - `LabelEncoder` for categorical fields
  - Numerical fields left as-is

### 2. **Dimensionality Reduction:**
- Used **TruncatedSVD** to reduce high-cardinality multi-hot matrices for:
  - Procedures (→ `svd_proc`)
  - Conditions (→ `svd_cond`)
  - Allergies (→ `svd_alg`)

### 3. **Modeling:**
- **Model Type:** `ClassifierChain` with `CatBoostClassifier` (GPU-accelerated)
- **Target:** Multi-label prediction for `MEDICATION_CODES`
- **Metrics:** Micro F1, Hamming Loss, Subset Accuracy
- **Hyperparameters Tuned:** Iterations, depth, learning rate, chain order

---

##  Model Inputs and Outputs

| Input Type        | Features                           |
|-------------------|------------------------------------|
| Categorical        | Gender, City, Race, Ethnicity, Marital, Encounter Code |
| Numerical          | Age, Height (cm), Weight (kg)     |
| Multi-label        | Procedures, Conditions, Allergies |
| Output             | One or more Medication Codes       |

---

##  Web App Features

### Form Inputs:
- Personal data: name, age, gender, race, etc.
- Clinical inputs: condition/procedure/encounter notes (with autocompletion)
- Allergy fields with code auto-fill using maps

### Real-time Prediction:
- Button to **predict medications**
- Predictions returned with **RxNorm drug names**

###  Submission Logging:
- Button to **submit patient record**
- Saves to `patient_records.csv` including prediction results

###  Autocomplete Integration:
- Uses **BioPortal API** for:
  - SNOMED Condition/Procedure/Encounter term suggestions
- Uses **RxNorm API** to fetch drug names from predicted codes

---

##  Notebooks

### `EDA.ipynb`
- Data import and cleaning
- Visualization of medication frequency, condition overlap, etc.
- Insight extraction and filtering based on top meds

### `FINAL_MODEL_TRAINING.ipynb`
- Feature engineering: encoding, SVD, merging features
- Model training with `CatBoostClassifier` inside `ClassifierChain`
- Evaluation on test split and saving `pkl` files for production

---

## 🏁 Running the App

###  Step 1: Install Requirements
```bash
pip install flask pandas scikit-learn catboost joblib requests awesomplete
```

### Step 2: Run Flask Server
```bash
python app.py
```

### Step 3: Open in Browser
Go to: [http://localhost:5000](http://localhost:5000)

---

## Key Highlights

-  Personalized, data-driven medication recommendation engine
-  Multi-label classification using CatBoost + Classifier Chains
- Interactive frontend with real-time suggestions via BioPortal
- RxNorm drug name resolution from predicted codes
- CSV logging for real-world record keeping

---

##  Future Improvements

- Integrate patient history or unstructured EHR notes using NLP
-  Use transformer-based models (e.g. T5, ClinicalBERT) for clinical note interpretation
- Connect to FHIR-compatible EMRs for real patient data
- Add analytics dashboards for prediction trends and accuracy
-  Move from joblib → ONNX model export for cross-platform deployment
