import csv
import json
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--input", help="Path to the input CSV file", default="./marfan_allnotes_022024.csv")

args = argparser.parse_args()

# Load the input CSV file
with open(args.input, "r") as f:
    reader = csv.DictReader(f)
    marfan_notes = list(reader)

note_list = []
train_list, val_list = [], []
train_case, train_control, val_case, val_control = 0, 0, 0, 0
for note_entry in marfan_notes:
    note_id, note, raw_label = note_entry['note_id'], note_entry['summary'], note_entry["label"]
    set_type, label = raw_label.split("_")
    note_list.append({
        'note_id': note_id,
        'summary': note,
        'label': label,
    })
    if set_type == "training":
        train_list.append(
            {
                'note_id': note_id,
                'summary': note,
                'label': label,
            }
        )
        if label == 'controls':
            train_control += 1
        else:
            train_case += 1
    else:
        val_list.append(
            {
                'note_id': note_id,
                'summary': note,
                'label': label,
            }
        )
        if label == 'controls':
            val_control += 1
        else:
            val_case += 1


print(train_case, train_control, val_case, val_control)
with open('marfan_notes.json', 'w') as f:
    json.dump(note_list, f, indent=4)

with open('marfan_train_notes.json', 'w') as f:
    json.dump(train_list, f, indent=4)

with open('marfan_val_notes.json', 'w') as f:
    json.dump(val_list, f, indent=4)