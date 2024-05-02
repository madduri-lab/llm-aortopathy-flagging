import json
import torch
import logging
import argparse
from config import eval_config
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from utils.model_utils import load_peft_model

parser = argparse.ArgumentParser()  
## model
parser.add_argument("--model_name", type=str, default="/scratch/bcdz/zl52/llama/7B")
parser.add_argument("--use_lora", type=str, default="True", choices=["True", "False"])
parser.add_argument("--lora_name", type=str, default="./model/marfan/pretrain.pt")

## data
parser.add_argument("--input_note_path", type=str, default="./data/datasets/raw/prompts_version1.json")
parser.add_argument("--output_response_path", type=str, default="./data/datasets/raw/response_version1.txt")

## evaluation
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--quantization", type=str, default="True", choices=["True", "False"])
parser.add_argument("--do_sample", type=str, default="True", choices=["True", "False"])
parser.add_argument("--use_cache", type=str, default="True", choices=["True", "False"])
parser.add_argument("--reproducible", type=str, default="True", choices=["True", "False"])
parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum numbers of tokens to generate.")
parser.add_argument("--length_penalty", type=float, default=1)
parser.add_argument("--temp", type=float, default=0.6)
parser.add_argument("--repetition_penalty", type=float, default=1)

args = parser.parse_args()

# Set up logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(args.output_response_path)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

eval_config.device = args.device
eval_config.quantization = True if args.quantization == "True" else False
eval_config.do_sample = True if args.do_sample == "True" else False
eval_config.use_cache = True if args.use_cache == "True" else False
eval_config.reproducible = True if args. reproducible == "True" else False
eval_config.max_new_tokens = args.max_new_tokens
eval_config.length_penalty = args.length_penalty
eval_config.temperature = args.temp
eval_config.repetition_penalty = args.repetition_penalty

if eval_config.reproducible:
    torch.cuda.manual_seed(eval_config.seed)
    torch.manual_seed(eval_config.seed)


if "mistral_large" not in args.lora_name:
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

notes = json.load(open(args.input_note_path))

with torch.no_grad():
    for note in notes:
        prompt, label = note['prmopt'], note['label']
        batch = tokenizer([prompt], return_tensors="pt")
        batch = {
            k: v.to(eval_config.device)
            for k, v in batch.items()
        }
        # outputs = model.generate(
        #     **batch,
        #     max_length=4096,
        #     # max_new_tokens=eval_config.max_new_tokens,
        #     do_sample=eval_config.do_sample,
        #     top_p=eval_config.top_p,
        #     temperature=eval_config.temperature,
        #     min_length=eval_config.min_length,
        #     use_cache=eval_config.use_cache,
        #     top_k=eval_config.top_k,
        #     repetition_penalty=eval_config.repetition_penalty,
        #     length_penalty=eval_config.length_penalty,
        # )

        outputs = model.generate(
            **batch,
            max_new_tokens=eval_config.max_new_tokens,
            do_sample=eval_config.do_sample,
            top_p=eval_config.top_p,
            temperature=eval_config.temperature,
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_without_input = prediction[len(prompt)-len("**Evaluation** "):]
        logger.info("================================================")
        prediction_post_process = prediction_without_input.split("=")[0].strip()
        logger.info(f"The label for the patient {note['note_id']} is {label}, and the total output length is {len(outputs[0])}")
        logger.info(prediction_post_process)
        logger.info("================================================")
        
