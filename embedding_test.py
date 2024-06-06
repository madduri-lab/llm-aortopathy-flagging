import os
import json
import torch
import pickle
import pathlib
import logging
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utils.model_utils import load_peft_model
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM

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
parser.add_argument("--lora_name", type=str, default="./model/marfan/mistral_small.pt")
parser.add_argument("--log_file", type=str, default="llama-embedding.txt")
parser.add_argument("--embedding_dir", type=str, default="./embedding/meditron-7b")
parser.add_argument("--embedding_type", type=str, default="mean", choices=["mean", "first", "last"])

## data
parser.add_argument("--train_note_path", type=str, default="./data/datasets/raw/marfan_train_notes.json")
parser.add_argument("--val_note_path", type=str, default="./data/datasets/raw/marfan_val_notes.json")

args = parser.parse_args()

# Set up embedding saving directory
if not os.path.exists(args.embedding_dir):
    pathlib.Path(args.embedding_dir).mkdir(parents=True, exist_ok=True)

# Set up logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(f'{args.embedding_dir}/{args.log_file}')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Load the model
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
else:
    logger.info("Use the original model")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
tokenizer.pad_token = tokenizer.bos_token

# Load data
cases = json.load(open('Aortopathy_cases_processed_cot.json'))
controls = json.load(open('Aortopathy_controls_processed_cot.json'))

embeddings = []
classes = []

with torch.no_grad():
    for i, note in enumerate(cases):
        prompt = note['Note']
        batch = tokenizer([prompt], return_tensors="pt")
        batch = {
            k: v.to(args.device)
            for k, v in batch.items()
        }
        output = model.forward(
            **batch,
            output_hidden_states=True,
        )
        if args.embedding_type == "mean":
            embedding = output.hidden_states[-1].squeeze().float().mean(0).cpu().detach().numpy()
        elif args.embedding_type == "first":
            embedding = output.hidden_states[-1].squeeze().float()[0].cpu().detach().numpy()
        elif args.embedding_type == "last":
            embedding = output.hidden_states[-1].squeeze().float()[-1].cpu().detach().numpy()
        if args.do_normalize == "True":
            embedding = embedding / np.linalg.norm(embedding, ord=2)
            print(sum(embedding**2))
        embeddings.append(embedding)
        classes.append(1)
    for i, note in enumerate(controls):
        prompt = note['Note']
        batch = tokenizer([prompt], return_tensors="pt")
        batch = {
            k: v.to(args.device)
            for k, v in batch.items()
        }
        output = model.forward(
            **batch,
            output_hidden_states=True,
        )
        if args.embedding_type == "mean":
            embedding = output.hidden_states[-1].squeeze().float().mean(0).cpu().detach().numpy()
        elif args.embedding_type == "first":
            embedding = output.hidden_states[-1].squeeze().float()[0].cpu().detach().numpy()
        elif args.embedding_type == "last":
            embedding = output.hidden_states[-1].squeeze().float()[-1].cpu().detach().numpy()
        if args.do_normalize == "True":
            embedding = embedding / np.linalg.norm(embedding, ord=2)
            print(sum(embedding**2))
        embeddings.append(embedding)
        classes.append(0)

X = np.concatenate([x[None] for x in embeddings], axis=0)
np.savez(f'{args.embedding_dir}/embedding_val_new.npz', embeddings=X)
with open(f'{args.embedding_dir}/classes_val_new.pkl', 'wb') as file:
    pickle.dump(classes, file)
