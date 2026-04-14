"""
Notes to few-shot prompts
"""

import json
import random
random.seed(42)


shot_examples = [
    {
        "note": "The patient, a male, was admitted due to fatigue and dyspnea on exertion. He has a history of a Bentall procedure with a mechanical valve, resection of aortic root pseudoaneurysm, Hepatitis C, hyperlipidemia, hemorrhoids, benign tremor, and seasonal allergies. He has never been tested for a genetic disorder that may cause physical features similar to his and his brother's. His brother also has dextrocardia but has not undergone genetic testing. Their father passed away from a stroke and their mother is currently living with dementia in an assisted living facility.\nThe patient's chief complaint was increasing shortness of breath over the past few months, to the point where he could not climb three flights of stairs. However, he did not report any chest pain, nausea/vomiting, or diaphoresis. He was seen by the cardiac surgery service and accepted for a redo Bentall procedure. Pre-op cardiac cath revealed a large ascending aorta pseudoaneurysm with proximity to the right coronary artery (RCA) take off. He was discharged to home on Lovenox and returned for a heparin bridge and redo sternotomy Bentall and coronary artery bypass graft (CABG) procedure.\nUpon admission, his vitals were stable and physical exam was largely unremarkable. His admission labs showed abnormal RBC count (3.89*), HGB (12.6*), HCT (36.0*), MCH (32.4*), and glucose (118*). His echocardiogram showed a mildly dilated right ventricular cavity with borderline normal systolic function, focal calcifications in the ascending aorta, mildly thickened mitral valve leaflets with moderate thickening of the chordae, and a pseudoaneurysm identified below the ascending aorta at the location of RCA implantation. His chest radiology report showed persistent right pleural effusion and likely right lower lobe atelectasis, unchanged right pleural effusion, improving left loculated pleural effusion, stable cardiomegaly, and prosthetic cardiac valve.\nDuring his hospital stay, he underwent repair of his pseudoaneurysm and tolerated the procedure well. He was extubated within several hours of arrival in the CVICU and weaned off of inotropic support on post-op day 1. His post-op course was largely uneventful and he was discharged to home on post-op day 7.\nUpon admission, he was on warfarin, fish oil, loratadine, multivitamin, and phosphatidyl choline.",
        "answer": "1. Clinical Features: Marfan syndrome is associated with a variety of clinical features, including cardiovascular manifestations such as aortic aneurysms and mitral valve abnormalities, as well as skeletal manifestations like tall stature and joint hypermobility. In this patient's case, they have a history of a Bentall procedure with a mechanical valve, resection of aortic root pseudoaneurysm, and evidence of a pseudoaneurysm below the ascending aorta at the location of the right coronary artery implantation. These findings raise suspicion for Marfan syndrome, especially considering the family history of dextrocardia in the patient's brother.\n2. Family History: Marfan syndrome has a genetic component and tends to run in families. The patient's brother has dextrocardia, which could potentially be related to a genetic disorder. Additionally, their father passed away from a stroke, which may or may not be related to Marfan syndrome, but could suggest a genetic predisposition to cardiovascular issues.\n3. Other Medical Conditions: It's also important to consider other medical conditions that may mimic or coexist with Marfan syndrome. The patient has a history of Hepatitis C, hyperlipidemia, hemorrhoids, benign tremor, and seasonal allergies, which are not typically associated with Marfan syndrome but should still be taken into account in the overall clinical assessment.\n4. Clinical Course: Assessing the patient's clinical course and response to treatment can also provide insights. In this case, the patient underwent repair of their pseudoaneurysm and had an uneventful post-operative course, which may suggest good overall health but does not necessarily rule out the possibility of an underlying genetic disorder like Marfan syndrome.\nConclusion: Based on these considerations, the patient's presentation, family history, and clinical course suggest that they should be considered a high priority for genetic testing for Marfan syndrome. This testing could provide valuable information for their diagnosis, prognosis, and management, particularly in the context of their cardiovascular issues and family history."
    },
    {
        "note": "The patient, a female, was admitted due to progressive pain in her right knee. The pain was diagnosed as being caused by osteoarthritis (OA) in the knee, as revealed by X-rays. Her past medical history includes borderline diabetes mellitus (DM) and gastritis. She has allergies to iodine, iodine-containing substances, silver, and mint.\nA major surgical procedure was performed on her right knee, specifically a total knee arthroplasty (TKA). The physical examination revealed a normal head, eyes, ears, nose, and throat (HEENT), clear lungs, regular rate and rhythm (RRR) in cardiovascular (CV) examination, and a soft, non-distended, non-tender abdomen with bowel sounds. However, her right knee was found to be varus with osteophytes.\nLab results showed several abnormal findings. Her white blood cell (WBC) count was elevated at 14.8, indicating a possible infection. Her red blood cell (RBC) count was low at 3.36, and her hemoglobin (HGB) and hematocrit (HCT) were also low at 10.2 and 30.6 respectively, suggesting anemia.\nDuring her stay, she was given prophylaxis for deep vein thrombosis (DVT) with boots and subcutaneous Lovenox. She also worked with a continuous passive motion (CPM) machine. Upon discharge, her condition was good and she was sent home with service. Her discharge medications included Oxycodone for pain, Docusate Sodium for potential constipation, and Enoxaparin for continued anticoagulation.\nThe patient was instructed to engage in activity as tolerated, use a walker or crutches for ambulation, and take her prescribed medications. She was also advised to turn, cough, and deep breathe every two hours when awake. Her surgical incision was dressed with dry gauze and she was told she could leave it open to air.",
        "answer": "1. Clinical Features: The patient's primary complaint and reason for admission is progressive pain in her right knee, ultimately diagnosed as osteoarthritis (OA). While OA is not typically associated with Marfan syndrome, it's important to consider if there are any other features suggestive of Marfan syndrome present in the patient, such as skeletal abnormalities, ocular manifestations, or cardiovascular issues. However, there is no indication of these features in the patient's history or physical examination.\n2. Family History: There is no mention of any family history suggestive of Marfan syndrome or related connective tissue disorders. Family history plays a crucial role in assessing the likelihood of a genetic condition like Marfan syndrome, as it often runs in families.\n3. Other Medical Conditions: The patient's past medical history includes borderline diabetes mellitus (DM) and gastritis, which are not typically associated with Marfan syndrome. Additionally, her allergies to iodine, silver, and mint are unrelated to Marfan syndrome.\n4. Clinical Course: The patient underwent a major surgical procedure, total knee arthroplasty (TKA), which is not directly related to Marfan syndrome. Her post-operative course included prophylaxis for deep vein thrombosis (DVT) and pain management with Oxycodone, which are standard procedures following joint replacement surgery. There is no indication of any complications related to connective tissue abnormalities or cardiovascular issues during her hospital stay.\nConclusion: Based on these considerations, there is no compelling evidence to suggest that this patient should undergo genetic testing for Marfan syndrome. The clinical features, family history, and medical course of the patient do not align with the typical presentation of Marfan syndrome. However, it's important to continue monitoring the patient's health and consider genetic testing if new symptoms or findings suggestive of Marfan syndrome arise in the future."
    }
]

queries = [
    "Above is a patient's medical progress note. You are a clinical expert on Marfan's Syndrome and an assistant to a clinician. Based on the note, summarize different evidence related to Marfan's Syndrome, and finally give the patient a designation of whether the patient: is high priority, low priority, or have no need for, genetic testing for Marfan's syndrome.",
]

marfan_notes = json.load(open('selected_notes.json'))
random.shuffle(marfan_notes)

for i, query in enumerate(queries):
    few_shot_template = ""
    for shot in shot_examples:
        few_shot_template += (
            "=========================================\n"
            f"Patient's medical progress note: {shot['note']}\n"
            "=========================================\n"
            f"Question: {query}\n"
            "=========================================\n"
            f"Answer: {shot['answer']}\n"
        )

    prompts = []
    for note_entry in marfan_notes:
        note, label = note_entry['summary'], note_entry['label']
        prompt = few_shot_template + (
            "=========================================\n"
            f"Patient's medical progress note: {note}\n"
            "=========================================\n"
            f"Question: {query}\n"
            "=========================================\n"
            "Answer: "
        )
        prompts.append({
            "prmopt": prompt,
            "label": label
        })
    with open(f'prompts_version{i+12}.json', 'w') as f:
        json.dump(prompts, f, indent=4)