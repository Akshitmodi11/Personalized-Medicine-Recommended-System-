from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import os
import joblib
import requests

app = Flask(__name__)

# Load static mappings for display (optional)
with open("static/condition_map.json") as f:
    cond_map = json.load(f)
with open("static/allergy_map.json") as f:
    allergy_map = json.load(f)

@app.route('/get_map/<map_type>')
def get_map(map_type):
    return jsonify({
        'condition': cond_map,
        'allergy': allergy_map
    }.get(map_type, {}))

# Load trained model and preprocessing artifacts
model = joblib.load("final model/medication_chain_catboost.pkl")
mlb_med = joblib.load("final model/mlb_medication (1).pkl")
mlb_proc = joblib.load("final model/mlb_procedure.pkl")
mlb_cond = joblib.load("final model/mlb_condition (1).pkl")
mlb_alg = joblib.load("final model/mlb_allergy (1).pkl")
label_encoders = joblib.load("final model/label_encoders (1).pkl")
svd_proc = joblib.load("final model/svd_proc.pkl")
svd_cond = joblib.load("final model/svd_cond.pkl")
svd_alg = joblib.load("final model/svd_alg.pkl")

csv_file_path = "patient_records.csv"

def split_note_and_code(note):
    if " — " in note:
        desc, code = note.split(" — ", 1)
        return desc.strip(), code.strip()
    return note.strip(), ""

def build_structured_input(form_data):
    procedure_codes = [split_note_and_code(x)[1] for x in form_data.getlist("procedure_notes") if x.strip()]
    condition_codes = [split_note_and_code(x)[1] for x in form_data.getlist("condition_notes") if x.strip()]
    allergy_codes = [split_note_and_code(x)[1] for x in form_data.getlist("allergies") if x.strip()]
    allergy_codes = [code.strip() for code in ','.join(allergy_codes).split(',') if code.strip().isdigit()]

    X_proc = svd_proc.transform(mlb_proc.transform([procedure_codes]))
    X_cond = svd_cond.transform(mlb_cond.transform([condition_codes]))
    X_alg = svd_alg.transform(mlb_alg.transform([allergy_codes]))

    cat_fields = ["BP_CITY", "MARITAL", "RACE", "ETHNICITY", "GENDER", "ENCOUNTER_CODE"]
    X_cat = []
    for col in cat_fields:
        val = form_data.get(col.lower(), '').strip()
        le = label_encoders[col]
        if val not in le.classes_:
            le.classes_ = np.append(le.classes_, val)
        X_cat.append(le.transform([val])[0])
    X_cat = np.array(X_cat).reshape(1, -1)

    try:
        age = float(form_data.get("birthdate", 0))
    except ValueError:
        age = 0
    try:
        weight = float(form_data.get("weight_kg", 0))
    except ValueError:
        weight = 0
    try:
        height = float(form_data.get("height_cm", 0))
    except ValueError:
        height = 0

    X_num = np.array([[age, weight, height]])
    return np.hstack([X_proc, X_cond, X_alg, X_cat, X_num])

def get_rxnorm_name(rxcui):
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
    response = requests.get(url)
    data = response.json()
    return data.get("properties", {}).get("name", f"Unknown RxCUI: {rxcui}")

def predict_medications(form_data):
    X = build_structured_input(form_data)
    y_pred = model.predict(X)
    meds = mlb_med.inverse_transform(y_pred)[0]
    med_name_pairs = [(code, get_rxnorm_name(code)) for code in meds]
    return med_name_pairs, meds

def save_patient_info(form_data, prediction_pairs, med_codes):
    def extract_notes(notes):
        parsed = [split_note_and_code(note) for note in notes if note.strip()]
        if not parsed:
            return [], []
        return zip(*parsed)

    encounter_desc, encounter_code = extract_notes(form_data.getlist('encounter_notes'))
    procedure_desc, procedure_code = extract_notes(form_data.getlist('procedure_notes'))
    condition_desc, condition_code = extract_notes(form_data.getlist('condition_notes'))
    allergy_desc, allergy_code = extract_notes(form_data.getlist('allergies'))

    record = {
        "First Name": form_data.get('first_name', ''),
        "Last Name": form_data.get('last_name', ''),
        "Gender": form_data.get('gender', ''),
        "Age": form_data.get('birthdate', ''),
        "Marital Status": form_data.get('marital', ''),
        "Race": form_data.get('race', ''),
        "Ethnicity": form_data.get('ethnicity', ''),
        "City": form_data.get('city', ''),
        "County": form_data.get('county', ''),
        "ZIP": form_data.get('zip', ''),
        "Smoking Status": form_data.get('smoking_status', ''),
        "Allergies Description": ', '.join(allergy_desc),
        "Allergies Code": ', '.join(allergy_code),
        "Height (cm)": form_data.get('height_cm', ''),
        "Weight (kg)": form_data.get('weight_kg', ''),
        "Encounter Description": ', '.join(encounter_desc),
        "Encounter Code": ', '.join(encounter_code),
        "Procedure Description": ', '.join(procedure_desc),
        "Procedure Code": ', '.join(procedure_code),
        "Condition Description": ', '.join(condition_desc),
        "Condition Code": ', '.join(condition_code),
        "Predicted Medications": ', '.join([f"{c} — {n}" for c, n in prediction_pairs]),
        "Predicted Medication Codes": ', '.join(med_codes)
    }

    df = pd.DataFrame([record])
    if os.path.exists(csv_file_path):
        existing = pd.read_csv(csv_file_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_file_path, index=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_pairs = None
    form_data = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        prediction_pairs, med_codes = predict_medications(form_data)
        if 'submit' in request.form:
            save_patient_info(form_data, prediction_pairs, med_codes)
            prediction_pairs = None  # Optional: clear prediction after saving

    return render_template('index.html', prediction=prediction_pairs, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
