"""
Notes to few-shot prompts
"""
import json

shot_examples = [
    {
        "note": "The patient, a male, was admitted due to fatigue and dyspnea on exertion. He has a history of a Bentall procedure with a mechanical valve, resection of aortic root pseudoaneurysm, Hepatitis C, hyperlipidemia, hemorrhoids, benign tremor, and seasonal allergies. He has never been tested for a genetic disorder that may cause physical features similar to his and his brother's. His brother also has dextrocardia but has not undergone genetic testing. Their father passed away from a stroke and their mother is currently living with dementia in an assisted living facility.\nThe patient's chief complaint was increasing shortness of breath over the past few months, to the point where he could not climb three flights of stairs. However, he did not report any chest pain, nausea/vomiting, or diaphoresis. He was seen by the cardiac surgery service and accepted for a redo Bentall procedure. Pre-op cardiac cath revealed a large ascending aorta pseudoaneurysm with proximity to the right coronary artery (RCA) take off. He was discharged to home on Lovenox and returned for a heparin bridge and redo sternotomy Bentall and coronary artery bypass graft (CABG) procedure.\nUpon admission, his vitals were stable and physical exam was largely unremarkable. His admission labs showed abnormal RBC count (3.89*), HGB (12.6*), HCT (36.0*), MCH (32.4*), and glucose (118*). His echocardiogram showed a mildly dilated right ventricular cavity with borderline normal systolic function, focal calcifications in the ascending aorta, mildly thickened mitral valve leaflets with moderate thickening of the chordae, and a pseudoaneurysm identified below the ascending aorta at the location of RCA implantation. His chest radiology report showed persistent right pleural effusion and likely right lower lobe atelectasis, unchanged right pleural effusion, improving left loculated pleural effusion, stable cardiomegaly, and prosthetic cardiac valve.\nDuring his hospital stay, he underwent repair of his pseudoaneurysm and tolerated the procedure well. He was extubated within several hours of arrival in the CVICU and weaned off of inotropic support on post-op day 1. His post-op course was largely uneventful and he was discharged to home on post-op day 7.\nUpon admission, he was on warfarin, fish oil, loratadine, multivitamin, and phosphatidyl choline.",
        "answer": "This patient is maybe likely to have Marfan syndrome. It is unclear what his age is, but aortic aneurysm requiring Bentall procedure at a young age would be possibly suspicious for Marfan syndrome or a related connective tissue disease. His brother’s history of dextrocardia is unusual and not seen in Marfan syndrome, and makes me suspicious that a different more complex genetic disorder should be considered. If he is under the age of 50, I would recommended genetic testing on a low priority basis."
    },
    {
        "note": "The patient, a female, was admitted due to progressive pain in her right knee. The pain was diagnosed as being caused by osteoarthritis (OA) in the knee, as revealed by X-rays. Her past medical history includes borderline diabetes mellitus (DM) and gastritis. She has allergies to iodine, iodine-containing substances, silver, and mint.\nA major surgical procedure was performed on her right knee, specifically a total knee arthroplasty (TKA). The physical examination revealed a normal head, eyes, ears, nose, and throat (HEENT), clear lungs, regular rate and rhythm (RRR) in cardiovascular (CV) examination, and a soft, non-distended, non-tender abdomen with bowel sounds. However, her right knee was found to be varus with osteophytes.\nLab results showed several abnormal findings. Her white blood cell (WBC) count was elevated at 14.8, indicating a possible infection. Her red blood cell (RBC) count was low at 3.36, and her hemoglobin (HGB) and hematocrit (HCT) were also low at 10.2 and 30.6 respectively, suggesting anemia.\nDuring her stay, she was given prophylaxis for deep vein thrombosis (DVT) with boots and subcutaneous Lovenox. She also worked with a continuous passive motion (CPM) machine. Upon discharge, her condition was good and she was sent home with service. Her discharge medications included Oxycodone for pain, Docusate Sodium for potential constipation, and Enoxaparin for continued anticoagulation.\nThe patient was instructed to engage in activity as tolerated, use a walker or crutches for ambulation, and take her prescribed medications. She was also advised to turn, cough, and deep breathe every two hours when awake. Her surgical incision was dressed with dry gauze and she was told she could leave it open to air.",
        "answer": "This patient has neither of the cardinal features of Marfan syndrome – aortic aneurysm or subluxation of the lens of the eye – and has no reported family history concerning of the syndrome. She is very unlikely to have Marfan syndrome and genetic testing is not needed."
    }
]

queries = [
    "**Instruction**: You are a clinical expert on Marfan's Syndrome and an assistant to a clinician. Based on the patient's medical progress note, give the patient a designation of whether they are high priority, low priority, or have no need for, genetic testing for Marfan's syndrome. Explain your thoughts.",
]

marfan_notes = json.load(open('selected_notes.json'))

for i, query in enumerate(queries):
    few_shot_template = ""
    for shot in shot_examples:
        few_shot_template += (
            "=============\n"
            f"{query}\n\n"
            f"**Patient's Medical Progress Note**: {shot['note']}\n\n"
            f"**Genetic Testing Evaluation**: {shot['answer']}\n"
        )

    prompts = []
    for note_entry in marfan_notes:
        note, label = note_entry['summary'], note_entry['label']
        note = note.replace("\n\n", "\n")
        prompt = few_shot_template + (
            "=============\n"
            f"{query}\n\n"
            f"**Patient's Medical Progress Note**: {note}\n\n"
            "**Genetic Testing Evaluation**: "
        )
        prompts.append({
            "prmopt": prompt,
            "label": label
        })
    with open(f'prompts_version{i+14}.json', 'w') as f:
        json.dump(prompts, f, indent=4)