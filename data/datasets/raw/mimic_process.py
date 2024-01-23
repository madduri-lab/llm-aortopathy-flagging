"""
This script extracts the clinical notes for marfan and non-marfan patients from  
the MIMIC dataset (https://physionet.org/content/mimic-iv-note/2.2).
"""

import os
import csv
import random

CASE_NOTE_IDS = [
    "10653756-DS-18",
    "10653756-DS-21",
    "10679870-DS-19",
    "12623596-DS-7",
    "12623596-DS-8",
    "13916274-DS-10",
    "13916274-DS-22",
    "16230458-DS-19",
    "17619493-DS-20",
    "17806530-DS-9",
    "18130295-DS-5",
    "18715851-DS-11"
]

CONTROL_NOTE_IDS = [
    "10760122-DS-12",
    "10760122-DS-14",
    "10760122-DS-15",
    "10760122-DS-16",
    "10760122-DS-17",
    "13039253-DS-18",
    "14063021-DS-13",
    "14596984-DS-14",
    "19953888-DS-11",
    "10004457-DS-10"
]

NOTE_DIRECTORY = "physionet.org/files/mimic-iv-note/2.2/note"
NOTE_FILE = "discharge"
SHUFFLE = True

input_file = os.path.join(NOTE_DIRECTORY, f"{NOTE_FILE}.csv")
output_case_file = os.path.join(NOTE_DIRECTORY, f"{NOTE_FILE}_marfan_case.csv")
output_control_file = os.path.join(NOTE_DIRECTORY, f"{NOTE_FILE}_marfan_control.csv")

output_case_rows = []
output_control_rows = []
with open(input_file, 'r', newline='') as csv_input_file:
    reader = csv.reader(csv_input_file)
    header = next(reader)
    for row in reader:
        if row[0] in CASE_NOTE_IDS:
            output_case_rows.append(row)
        elif row[0] in CONTROL_NOTE_IDS:
            output_control_rows.append(row)

if SHUFFLE:
    random.seed(1)
    random.shuffle(output_case_rows)
    random.shuffle(output_control_rows)

with open(output_case_file, 'w', newline='') as csv_output_file:
    writer = csv.writer(csv_output_file)
    writer.writerows(output_case_rows)

with open(output_control_file, 'w', newline='') as csv_output_file:
    writer = csv.writer(csv_output_file)
    writer.writerows(output_control_rows)