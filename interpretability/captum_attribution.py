import time
import torch
import argparse
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import (
    FeatureAblation, 
    LLMAttribution, 
    TextTokenInput, 
)

def load_peft_model(model, peft_model_path):
    """Load the saved peft model into the based model."""
    peft_state_dict = torch.load(peft_model_path, map_location="cuda")
    model.load_state_dict(peft_state_dict, strict=False)
    return model

parser = argparse.ArgumentParser()  
## model
parser.add_argument(
    "--model_name", 
    type=str, 
    default="meta-llama/Meta-Llama-3-8B", 
    choices=[
        "epfl-llm/meditron-7b",
        "epfl-llm/meditron-70b",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Meta-Llama-3-8B", 
        "meta-llama/Meta-Llama-3-70B",
        "mistralai/Mistral-7B-v0.1", 
        "mistralai/Mixtral-8x7B-v0.1",
        "mistralai/Mixtral-8x22B-v0.1",
    ]
)
parser.add_argument("--use_lora", type=str, default="True", choices=["True", "False"])
parser.add_argument("--lora_name", type=str, default="./model/marfan/llama3_8b_genrev_aora_raw_large.pt")
parser.add_argument("--device", type=str, default="cuda")

args = parser.parse_args()

# Load the model
if "large" not in args.lora_name:
    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        r = 16,
        lora_alpha = 32,
        target_modules = ["q_proj", "v_proj"],
        bias= "none",
        lora_dropout = 0.05,
    )
else:
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    return_dict=True,
    load_in_8bit=True,
    device_map="auto",
    low_cpu_mem_usage=True,
)

if args.use_lora == "True":
    model = get_peft_model(model, lora_config)
    model = load_peft_model(model, args.lora_name)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
tokenizer.pad_token = tokenizer.bos_token
start = time.time()
eval_prompt = """**Instruction**: You are a clinical expert on rare genetic diseases. Does this patient need genetic testing for a potential undiagnosed rare genetic disease or syndrome based on their past and present symptoms and medical history? If yes, provide specific criteria why, otherwise state why the patient does not require genetic testing. Return your response as a JSON formatted string with two parts 1) testing recommendation ('recommended' or 'not recommended') and 2) your reasoning.

**Patient's Medical Progress Note**: This is a clinical note for a patient 'Discharge Summary:  Patient Name: [Redacted] DOB: [Redacted] Admission Date: [Redacted] Discharge Date: [Redacted]  Hospital Course:  The patient was admitted due to deterioration of his pre-existing medical conditions, including pulmonary artery aneurysm, severe pulmonic regurgitation, and biventricular failure with an ejection fraction of 22%. The patient had undergone pulmonary artery resection with pulmonary homograft valve 27 mm implantation, resulting in significant improvement of his symptoms and an improved ejection fraction of 45%.  During the hospital stay, the patient developed acute respiratory distress, shortness of breath, and fever. He was found to have contracted Klebsiella pneumonia, which led to multiorgan system failure, including acute liver failure, acute renal failure, and respiratory failure requiring ventilator support. He was also diagnosed with acute-on-chronic systolic heart failure and pulmonary artery hypertension.  After the patient's multiorgan system failure had subsided, he was discharged from the hospital. He was later found to have reduced left lung volume with restrictive disease of unknown etiology.  Diagnosis:  The patient's medical history was suggestive of an underlying connective tissue disorder. His family history includes cardiomyopathy in three paternal uncles, father, and sister.  Treatment:  The patient was considered for a percutaneous pulmonary valve replacement or some form of mechanical circulatory support including a left ventricular assist device (LVAD), biventricular assist device (BIVAD), or total artificial heart while he waited for a transplant.  Outcome:  The patient's pulmonary valve regurgitation and pulmonary artery aneurysm worsened since his surgery at age 35, with the aneurysm reaching 5.2 cm. These symptoms prompted his placement on the cardiac transplantation list. He has been closely followed up by the cardiology clinic, and treatment options are being considered.  Follow Up:  The patient will continue to follow up with the cardiology clinic as well as other specialists as needed to monitor his condition and consider further treatment options.  Diet:  The patient is recommended to follow a healthy and balanced diet and to avoid food and drinks that negatively impact his medical condition.  Activity:  The patient is advised to engage in regular, light physical activity as much as possible, with specific exercises as recommended by his healthcare provider.  Medications:  The patient is prescribed medications as needed to maintain his medical conditions, and his medication list is available in the medical records.

**Evaluation**: 
"""
fa = FeatureAblation(model)
llm_attr = LLMAttribution(fa, tokenizer)

inp = TextTokenInput(
    eval_prompt, 
    tokenizer,
    skip_tokens=[1],  # skip the special token for the start of the text <s>
)

target = """{
    "designation": "recommended",
    "reasoning": "Given the patient's complex medical history and family background, genetic testing for aortopathy-related diseases or syndromes is strongly recommended. The patient has a significant history of severe cardiovascular issues, including a pulmonary artery aneurysm, severe pulmonic regurgitation, and biventricular failure, all of which suggest a potential underlying genetic predisposition. Additionally, the presence of cardiomyopathy in multiple paternal relatives further supports the likelihood of a hereditary connective tissue disorder. Identifying a genetic cause could provide critical insights into the patient's condition, guide personalized treatment strategies, and inform family members about their own risks and necessary preventive measures."
}"""

attr_res = llm_attr.attribute(inp, target=target)
print("attr to the output sequence:", attr_res.seq_attr.shape)  # shape(n_input_token)
print("attr to the output tokens:", attr_res.token_attr.shape)
np.save('sq_attr_raw.npy', attr_res.seq_attr.cpu().numpy())
np.save('token_attr_raw.npy', attr_res.token_attr.cpu().numpy())
print(time.time()-start)
