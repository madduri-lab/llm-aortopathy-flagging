import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

argparser = argparse.ArgumentParser()
argparser.add_argument('--temperature', type=float, default=8)
argparser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
argparser.add_argument('--sq_attr_path', type=str, default='sq_attr_raw.npy')
argparser.add_argument('--output_name', type=str, default='test_name')
args = argparser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.pad_token = tokenizer.bos_token

eval_prompt = """**Instruction**: You are a clinical expert on rare genetic diseases. Does this patient need genetic testing for a potential undiagnosed rare genetic disease or syndrome based on their past and present symptoms and medical history? If yes, provide specific criteria why, otherwise state why the patient does not require genetic testing. Return your response as a JSON formatted string with two parts 1) testing recommendation ('recommended' or 'not recommended') and 2) your reasoning.

**Patient's Medical Progress Note**: This is a clinical note for a patient 'Discharge Summary:  Patient Name: [Redacted] DOB: [Redacted] Admission Date: [Redacted] Discharge Date: [Redacted]  Hospital Course:  The patient was admitted due to deterioration of his pre-existing medical conditions, including pulmonary artery aneurysm, severe pulmonic regurgitation, and biventricular failure with an ejection fraction of 22%. The patient had undergone pulmonary artery resection with pulmonary homograft valve 27 mm implantation, resulting in significant improvement of his symptoms and an improved ejection fraction of 45%.  During the hospital stay, the patient developed acute respiratory distress, shortness of breath, and fever. He was found to have contracted Klebsiella pneumonia, which led to multiorgan system failure, including acute liver failure, acute renal failure, and respiratory failure requiring ventilator support. He was also diagnosed with acute-on-chronic systolic heart failure and pulmonary artery hypertension.  After the patient's multiorgan system failure had subsided, he was discharged from the hospital. He was later found to have reduced left lung volume with restrictive disease of unknown etiology.  Diagnosis:  The patient's medical history was suggestive of an underlying connective tissue disorder. His family history includes cardiomyopathy in three paternal uncles, father, and sister.  Treatment:  The patient was considered for a percutaneous pulmonary valve replacement or some form of mechanical circulatory support including a left ventricular assist device (LVAD), biventricular assist device (BIVAD), or total artificial heart while he waited for a transplant.  Outcome:  The patient's pulmonary valve regurgitation and pulmonary artery aneurysm worsened since his surgery at age 35, with the aneurysm reaching 5.2 cm. These symptoms prompted his placement on the cardiac transplantation list. He has been closely followed up by the cardiology clinic, and treatment options are being considered.  Follow Up:  The patient will continue to follow up with the cardiology clinic as well as other specialists as needed to monitor his condition and consider further treatment options.  Diet:  The patient is recommended to follow a healthy and balanced diet and to avoid food and drinks that negatively impact his medical condition.  Activity:  The patient is advised to engage in regular, light physical activity as much as possible, with specific exercises as recommended by his healthcare provider.  Medications:  The patient is prescribed medications as needed to maintain his medical conditions, and his medication list is available in the medical records.

**Evaluation**: 
"""

# Generate tokens for the evaluation prompt and target
eval_prompt_token = tokenizer(eval_prompt, return_tensors='pt')['input_ids']
eval_prompt_tokens = tokenizer.convert_ids_to_tokens(eval_prompt_token[0].tolist())

# Process the sequence attention scores
sq_attr = np.load(args.sq_attr_path)
sq_token_processed = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in eval_prompt_tokens]
sq_attr = [np.exp(attr / args.temperature) for attr in sq_attr]
max_attr = max(sq_attr)
sq_attr = [attr / max_attr for attr in sq_attr]
sq_colors = [plt.cm.Blues(value) for value in sq_attr]

sq_html_output = ''.join(f'<span style="background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]});">{token}</span>' for token, color in zip(sq_token_processed, sq_colors))
sq_html_output = '<p>' + sq_html_output + '</p>'

with open(f'{args.output_name}.html', 'w') as f:
    f.write(sq_html_output)
