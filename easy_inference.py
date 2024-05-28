import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from utils.model_utils import load_peft_model

LOG_FILE = "inference.log"
INPUT_FILE = "interface.txt"

parser = argparse.ArgumentParser()  
## model
parser.add_argument(
    "--model_name", 
    type=str, 
    default="meta-llama/Llama-2-7b-hf", 
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
parser.add_argument(
    "--lora_name", 
    type=str, 
    default="./model/marfan/llama2_7b_large.pt",
    choices=[
        "./model/marfan/llama2_7b_large.pt",
        "./model/marfan/llama2_7b_small.pt",
        "./model/marfan/llama2_13b_large.pt",
        "./model/marfan/llama3_8b_large.pt",
        "./model/marfan/llama3_8b_small.pt",
        "./model/marfan/meditron_7b_large.pt",
        "./model/marfan/meditron_7b_small.pt",
        "./model/marfan/mistral_large.pt",
        "./model/marfan/mistral_small.pt",
    ]
)
parser.add_argument("--device", type=str, default="cuda")

args = parser.parse_args()

def log(content: str):
    with open(LOG_FILE, 'a') as file:
        file.write(content)

def get_input(
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 100,
    echo: bool = True,
):
    input(f"Put your input in the file {INPUT_FILE} and press Enter to continue")
    with open(INPUT_FILE, 'r') as f:
        prompt = f.read()
    # ask user to input temperature, top_p, max_new_tokens
    temperature = float(input(f"Enter temperature: [Current value is {temperature} ] Press Enter to keep the current value: ") or temperature)
    top_p = float(input(f"Enter top_p: [Current value is {top_p} ] Press Enter to keep the current value: ") or top_p)
    max_new_tokens = int(input(f"Enter max_new_tokens: [Current value is {max_new_tokens} ] Press Enter to keep the current value: ") or max_new_tokens)
    echo = input(f"Do you want to echo the input or not? [Current choice is {echo} ] Enter 1 for echo and other for not.")
    echo = echo == "1"
    return prompt, temperature, top_p, max_new_tokens, echo

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

temperature, top_p, max_new_tokens, echo = 1.0, 1.0, 100, False
with torch.no_grad():
    while True:
        prompt, temperature, top_p, max_new_tokens, echo = get_input(temperature, top_p, max_new_tokens, echo)
        tokenized_prompt = tokenizer([prompt], return_tensors="pt")
        tokenized_prompt = {
            k: v.to(args.device)
            for k, v in tokenized_prompt.items()
        }
        outputs = model.generate(
            **tokenized_prompt,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            do_sample=True
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not echo:
            prediction = prediction[len(prompt):]
            # Post-process the prediction
            prediction = prediction.split("==")[0].strip()
        prediction_print = f"#####################################################\n{prediction}\n#####################################################\n"
        log(prediction_print)
        print(prediction_print)
