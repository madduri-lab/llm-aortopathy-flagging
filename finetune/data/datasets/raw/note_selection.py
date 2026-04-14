import csv
import json
import torch
import random
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
argparser.add_argument("--input", help="Path to the input CSV file", default="./marfan_allnotes_022024.csv")

args = argparser.parse_args()

# Load the input CSV file
with open(args.input, "r") as f:
    reader = csv.DictReader(f)
    marfan_notes = list(reader)

random.seed(42)
random.shuffle(marfan_notes)

case_list, control_list = [], []
patient_ids = []

for note_entry in marfan_notes:
    note_id, note, raw_label = note_entry['note_id'], note_entry['summary'], note_entry["label"]
    set_type, label = raw_label.split("_")
    patient_id = note_id.split('-')[0]
    if patient_id not in patient_ids:
        patient_ids.append(patient_id)
        if label == 'controls' and len(control_list) < 10:
            control_list.append({
                'note_id': note_id,
                'summary': note,
                'label': label
            })
        if label == 'cases' and len(case_list) < 10:
            case_list.append({
                'note_id': note_id,
                'summary': note,
                'label': label
            })
    if len(case_list) == 10 and len(control_list) == 10:
        break

print("Case Note ID:")
for case in case_list:
    print(case['note_id'])

print("Control Note ID:")
for control in control_list:
    print(control['note_id'])

all_list = case_list + control_list
with open('selected_notes.json', 'w') as f:
    json.dump(all_list, f, indent=4)