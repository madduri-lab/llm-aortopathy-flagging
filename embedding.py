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
parser.add_argument("--cluster_file", type=str, default="meditron-7b.pdf")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--embedding_type", type=str, default="mean", choices=["mean", "first", "last"])
parser.add_argument("--do_normalize", type=str, choices=["True", "False"], default="False")
parser.add_argument("--do_standardize", type=str, choices=["True", "False"], default="True")

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
train_notes = json.load(open(args.train_note_path))
val_notes = json.load(open(args.val_note_path))
logger.info(f"Train set length: {len(train_notes)}; Val set length: {len(val_notes)}")

train_embeddings, val_embeddings = [], []
train_classes, val_classes = [], []

# Generate embeddings
with torch.no_grad():
    for i, note in enumerate(train_notes):
        logger.info(i)
        prompt, label = note['summary'], note['label']
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
        train_embeddings.append(embedding)
        train_classes.append(
            0 if label == "controls" else 1
        )

    for i, note in enumerate(val_notes):
        logger.info(i)
        prompt, label = note['summary'], note['label']
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
        val_embeddings.append(embedding)
        val_classes.append(
            0 if label == "controls" else 1
        )

all_embeddings = train_embeddings + val_embeddings
all_classes = train_classes + val_classes

X_train = np.concatenate([x[None] for x in train_embeddings], axis=0)
X_val = np.concatenate([x[None] for x in val_embeddings], axis=0) 
X = np.concatenate([x[None] for x in all_embeddings], axis=0) 

# Save embeddings and classes
np.savez(f'{args.embedding_dir}/train_embedding.npz', train_embeddings=X_train)
np.savez(f'{args.embedding_dir}/val_embedding.npz', val_embeddings=X_val)
np.savez(f'{args.embedding_dir}/all_embedding.npz', all_embeddings=X)
with open(f'{args.embedding_dir}/train_classes.pkl', 'wb') as file:
    pickle.dump(train_classes, file)
with open(f'{args.embedding_dir}/val_classes.pkl', 'wb') as file:
    pickle.dump(val_classes, file)
with open(f'{args.embedding_dir}/all_classes.pkl', 'wb') as file:
    pickle.dump(all_classes, file)

# TSNE dimention reduction visualization
reduced = TSNE(n_components=2, random_state=0).fit_transform(X)
plt.scatter(reduced[:, 0], reduced[:, 1], c=all_classes, cmap='rainbow')
plt.savefig(f"{args.embedding_dir}/{args.cluster_file}")

if args.do_standardize == "True":
    scaler = StandardScaler()
    train_x = scaler.fit_transform(X_train)
    test_x = scaler.transform(X_val)
else:
    train_x = X_train
    test_x = X_val

# For a real problem, C should be properly cross validated and the confusion matrix analyzed
clf = LogisticRegression(random_state=0, C=1.0, max_iter=1000).fit(train_x, train_classes) 
prediction = clf.predict(test_x)
target = val_classes
logger.info(f"Precision: {100*np.mean(prediction == target):.2f}%")
cm = confusion_matrix(prediction, target)
logger.info(f"Confusion matrix is {cm}")
