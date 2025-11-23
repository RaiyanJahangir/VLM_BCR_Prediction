import pandas as pd
import os
import json
from datetime import datetime
from collections import defaultdict
import math

# ========== CLEANING HELPER ==========
def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

# ========== LOAD DATA ==========
metadata_df = pd.read_csv(
    "/mnt/data1/raiyan/breast_cancer/datasets/embed/tables/EMBED_OpenData_metadata.csv",
    low_memory=False
)
clinical_df = pd.read_csv(
    "/mnt/data1/raiyan/breast_cancer/datasets/embed/tables/EMBED_OpenData_clinical.csv",
    low_memory=False
)

# Drop unnamed columns
metadata_df = metadata_df.loc[:, ~metadata_df.columns.str.startswith("Unnamed")]
clinical_df = clinical_df.loc[:, ~clinical_df.columns.str.startswith("Unnamed")]

# Create full list of clinical fields (after dropping unnamed columns)
all_clinical_cols = clinical_df.columns.tolist()

# ========== CLINICAL DICT WITH MULTIPLE ENTRIES ==========
clinical_dict = defaultdict(list)
for _, row in clinical_df.iterrows():
    key = (row["empi_anon"], row["acc_anon"])
    clinical_dict[key].append(row.to_dict())

# ========== GROUP METADATA BY PATIENT ==========
patients = defaultdict(list)
for _, row in metadata_df.iterrows():
    empi = row["empi_anon"]
    patients[empi].append(row)

# ========== HELPER: PNG PATH RESOLUTION ==========
def find_png_path(filename, cohort):
    base_path = f"/mnt/data1/raiyan/breast_cancer/datasets/embed/png_images/cohort_{cohort}"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# ========== OUTPUT DIR ==========
output_dir = "/mnt/data1/raiyan/breast_cancer/datasets/embed/per_patient_json"
os.makedirs(output_dir, exist_ok=True)

# ========== MAIN LOOP ==========
for i, (empi, rows) in enumerate(patients.items(), 1):
    patient_dict = {"empi_anon": empi, "exams": {}}
    sorted_rows = sorted(rows, key=lambda x: x.get("study_date_anon", ""))
    
    for row in sorted_rows:
        acc = row["acc_anon"]
        study_date = row["study_date_anon"]
        cohort = row["cohort_num"]
        exam_key = f"{study_date}_{acc}"

        # Build clinical entries (list of all records)
        raw_clinicals = clinical_dict.get((empi, acc), [])
        clinical_entries = []
        for raw in raw_clinicals:
            clinical_entry = {col: raw.get(col, None) for col in all_clinical_cols}
            clinical_entries.append(clinical_entry)

        # Resolve PNG path
        png_filename = row["png_filename"]
        png_path = find_png_path(png_filename, cohort)
        if not png_path:
            continue  # Skip if PNG not found

        # Create or update exam entry
        if exam_key not in patient_dict["exams"]:
            patient_dict["exams"][exam_key] = {
                "study_date": study_date,
                "acc_anon": acc,
                "clinical": clinical_entries,
                "images": [],
            }

        # Add image info
        image_info = {
            "png_path": png_path,
            "image_type": row.get("FinalImageType"),
            "laterality": row.get("ImageLateralityFinal"),
            "view_position": row.get("ViewPosition"),
            "category": row.get("category"),
        }
        patient_dict["exams"][exam_key]["images"].append(image_info)

    # Skip if no exams
    if not patient_dict["exams"]:
        continue

    # Save JSON
    output_path = os.path.join(output_dir, f"{empi}.json")
    with open(output_path, "w") as f:
        json.dump(clean_nans(patient_dict), f, indent=2, default=str)

    if i % 20 == 0:
        print(f"Processed {i} patients")

print("Processing complete.")
