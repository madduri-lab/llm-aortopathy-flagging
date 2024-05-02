import csv
import json
import torch
import argparse

queries = [
    "Above is a patient's progress note. You are a clinical expert on marfan's syndrome and an assistant to a clinician. Based on the note, what is the likelihood that this patient has marfan's syndrome? Please explain why.",
    "Above is a patient's progress note. You are a clinical expert on marfan's syndrome and an assistant to a clinician. Based on the note, what is the likelihood this patient has marfan's syndrome: unlikely, maybe likely, likely, very likely, almost certainly?",
    "Above is a patient's progress note. You are a clinical expert on marfan's syndrome and an assistant to a clinician. Based on the note, what is the likelihood this patient has marfan's syndrome: unlikely, maybe likely, likely, very likely, almost certainly? Please explain why.",
    "Above is a patient's progress note. You are a clinical expert on marfan's syndrome and an assistant to a clinician. Based on the note, generate a quantitative score from 1-10 representing the likelihood this patient has marfan's syndrome? 1 is very unlikely, 10 is almost certainly. Please explain why you gave the score you did.",
    "Above is a patient's progress note. You are a clinical expert on marfan's syndrome and an assistant to a clinician. Based on the note, should this patient receive genetic testing for marfan's syndrome?",
    "Above is a patient's progress note. You are a clinical expert on marfan's syndrome and an assistant to a clinician. Based on the note, should this patient receive genetic testing for marfan's syndrome? Please explain why.",
    "Above is a patient's progress note. You are a clinical expert on marfan's syndrome and an assistant to a clinician. Based on the note, provide quantitative measurement from 1-10 where 1 is lowest and 10 is highest risk for marfan's syndrome. Please explain why and address the following symptom areas in your explanation: cardiovascular symptoms, physical appearance and disposition including skin and/or skeletal symptoms, ocular symptoms, family history or genetic data, nervous system and/or cognitive symptoms. Using these symptom areas in addition to any others you see fit to generate a single score."
]

argparser = argparse.ArgumentParser()
argparser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
argparser.add_argument("--input", help="Path to the input CSV file", default="./marfan_allnotes_022024.csv")

args = argparser.parse_args()

# Load the input CSV file
with open(args.input, "r") as f:
    reader = csv.DictReader(f)
    marfan_notes = list(reader)

for i, query in enumerate(queries):
    prompts = []
    for note_entry in marfan_notes:
        note, raw_label = note_entry['summary'], note_entry['label']
        set_type, label = raw_label.split("_")
        prompt = (
            "=========================================\n"
            f"{note}\n"
            "=========================================\n"
            f"{query}\n"
            "=========================================\n"
            "Answer: "
        )
        prompts.append({
            "prmopt": prompt,
            "label": label
        })
        with open(f'prompts_version{i+1}.json', 'w') as f:
            json.dump(prompts, f, indent=4)