import json
import random
random.seed(42)

queries = [
    "Above is a patient's medical progress note. You are a clinical expert on Marfan's Syndrome and an assistant to a clinician. Based on the note, explain how you know whether the patient is likely to have Marfan's Syndrome or not. I want to know the clinical intuition. Then give the patient a designation of likelihood: highly unlikely, unlikely, maybe likely, likely, very likely, almost certainly.",
    "Above is a patient's medical progress note. You are a clinical expert on Marfan's Syndrome and an assistant to a clinician. Based on the note, explain how you know whether the patient should be genetically tested or not for Marfan's Syndrome. I want to know the clinical intuition."
]

marfan_notes = json.load(open('selected_notes.json'))
random.shuffle(marfan_notes)

for i, query in enumerate(queries):
    prompts = []
    for note_entry in marfan_notes:
        note, label = note_entry['summary'], note_entry['label']
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
    with open(f'prompts_version{i+8}.json', 'w') as f:
        json.dump(prompts, f, indent=4)