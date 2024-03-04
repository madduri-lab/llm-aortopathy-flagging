import torch
import pickle
import logging
import argparse
from utils.model_utils import load_model, load_peft_model
from peft import get_peft_model, LoraConfig, TaskType
from transformers import LlamaTokenizer, default_data_collator
from config import eval_config
from data import ClinicalNoteDataset

parser = argparse.ArgumentParser()  
## model
parser.add_argument("--model_name", type=str, default="/scratch/bcdz/zl52/llama/7B")
parser.add_argument("--use_lora", type=str, default="True", choices=["True", "False"])
parser.add_argument("--lora_name", type=str, default="./model/marfan_prediction/lora_7B.pt")
parser.add_argument("--log_file", type=str, default="eval.log")

## data
parser.add_argument("--eval_data_path", type=str, default="./data/datasets/rag/marfan_rag_test.json")

## evaluation
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--quantization", type=str, default="True", choices=["True", "False"])
parser.add_argument("--do_sample", type=str, default="True", choices=["True", "False"])
parser.add_argument("--use_cache", type=str, default="True", choices=["True", "False"])
parser.add_argument("--reproducible", type=str, default="True", choices=["True", "False"])
parser.add_argument("--max_new_tokens", type=int, default=1, help="Maximum numbers of tokens to generate.")
parser.add_argument("--output_filename", type=str, default="result.txt")

args = parser.parse_args()

logging.basicConfig(filename=args.log_file, 
                    filemode='a', # 'a' to append, 'w' to overwrite
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

eval_config.device = args.device
eval_config.batchsize = args.batchsize
eval_config.quantization = True if args.quantization == "True" else False
eval_config.do_sample = True if args.do_sample == "True" else False
eval_config.use_cache = True if args.use_cache == "True" else False
eval_config.reproducible = True if args. reproducible == "True" else False
eval_config.max_new_tokens = args.max_new_tokens

if eval_config.reproducible:
    torch.cuda.manual_seed(eval_config.seed)
    torch.manual_seed(eval_config.seed)

lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r = 16,
    lora_alpha = 32,
    target_modules = ["q_proj", "v_proj"],
    bias= "none",
    lora_dropout = 0.05,
    inference_mode = False
)

model = load_model(args.model_name, quantization=True)
if args.use_lora == "True":
    model = get_peft_model(model, lora_config)
    model = load_peft_model(model, args.lora_name)
else:
    print("Use the original model")
model.eval()

tokenizer = LlamaTokenizer.from_pretrained(args.model_name, padding_side="left")
tokenizer.pad_token = tokenizer.bos_token

eval_dataset = ClinicalNoteDataset(args.eval_data_path, tokenizer, inference=True)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=eval_config.batchsize,
    num_workers=1,
    pin_memory=True,
    shuffle=True,
    collate_fn=default_data_collator,
)

results = {
    "cases": [],
    "controls": [],
    "unknown": []
}

def decode_label(label, tokenizer):
    i = 0
    while i < len(label) and label[i] == -100:
        i += 1
    label = label[i:-1]
    return tokenizer.decode(label, skip_special_tokens=True)

def extract_answer(text):
    last_index = text.rfind("Answer:")  
    if last_index != -1:
        return text[last_index + len("Answer:"):].strip() 
    else:
        return "" 

with torch.no_grad():
    for batch in eval_dataloader:
        batch = {
            k: v.to(eval_config.device) 
            for k, v in batch.items()
        }
        labels = list(batch["labels"])
        del batch["labels"]
        labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        outputs = model.generate(
            **batch,
            max_new_tokens=eval_config.max_new_tokens,
            do_sample=eval_config.do_sample,
            top_p=eval_config.top_p,
            temperature=eval_config.temperature,
            min_length=eval_config.min_length,
            use_cache=eval_config.use_cache,
            top_k=eval_config.top_k,
            repetition_penalty=eval_config.repetition_penalty,
            length_penalty=eval_config.length_penalty,
        )
        for i in range(eval_config.batchsize):
            prediction = tokenizer.decode(outputs[i], skip_special_tokens=True)
            prediction = extract_answer(prediction)
            logging.info(f"Target: {labels[i]} Prediction: {prediction}")
            if labels[i] == "cases":
                if "case" in prediction:
                    results["cases"].append(1)
                elif "control" in prediction:
                    results["cases"].append(0)
                else:
                    results["unknown"].append(prediction)
            else:
                if "case" in prediction:
                    results["controls"].append(0)
                elif "control" in prediction:
                    results["controls"].append(1)
                else:
                    results["unknown"].append(prediction)

logging.info(results)
with open(args.output_filename, 'wb') as file:
    pickle.dump(results, file)