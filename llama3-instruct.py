import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from utils.model_utils import load_peft_model

parser = argparse.ArgumentParser()  
## model
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--use_lora", type=str, default="True", choices=["True", "False"])
parser.add_argument("--lora_name", type=str, default="./model/marfan/llama3_instruct_8b_genrev_aora_raw_large.pt")
parser.add_argument("--device", type=str, default="cuda")


## data
parser.add_argument("--eval_data_path", type=str, default="./test-instruct.json")

args = parser.parse_args()

if "large" not in args.lora_name:
    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        r = 16,
        lora_alpha = 32,
        target_modules = ["q_proj", "v_proj"],
        bias= "none",
        lora_dropout = 0.05,
        inference_mode = False
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
        lora_dropout=0.05,  # Conventional
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
else:
    print("Use the original model")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
tokenizer.pad_token = tokenizer.bos_token

input_json = json.load(open(args.eval_data_path))

system_prompt = "You are a clinical expert on rare genetic diseases. Does this patient need genetic testing  for a potential undiagnosed rare genetic disease or syndrome based on their past and present symptoms and medical history? If yes, provide specific criteria why, otherwise state why the patient does not require genetic testing. Return your response as a JSON formatted string with two parts 1) testing recommendation ('recommendeded' or 'not recommended') and 2) your reasoning."

with torch.no_grad():
    for note in input_json:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": note['prompt']}
        ]
        non_token = tokenizer.apply_chat_template(
            chat,
            tokenize=False
        )
        tokens = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        print(tokens)
        tokens = tokens.to(args.device)
        outputs = model.generate(
            input_ids=tokens,
            max_new_tokens=300,
            top_p=1.0,
            temperature=0.6,
            do_sample=True,
            use_cache=False,
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=False)
        prediction = prediction[len(non_token):]
        print(prediction)
        print('============')



chat = [
   {"role": "user", "content": "Hello, how are you?"},
   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

print(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
