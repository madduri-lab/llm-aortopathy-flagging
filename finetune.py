import torch
import argparse
from data import AlpacaDataset
from train import train
from config import train_config
from utils.model_utils import load_model, save_peft_model
from transformers import LlamaTokenizer, default_data_collator
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import time

parser = argparse.ArgumentParser()  
parser.add_argument("--model_name", type=str, default="/scratch/bcdz/zl52/llama/7B")
parser.add_argument("--train_data_path", type=str, default="./data/raw_data/data_cb.json")
parser.add_argument("--validation_data_path", type=str, default="./data/raw_data/data_cb_val.json")
parser.add_argument("--output_name", type=str, default="./model/cb/lora_7B.pt")
parser.add_argument("--max_words", type=int, default=224, help="Maximum number of tokens for the input.")
parser.add_argument("--batch_size_training", type=int, default=4)
parser.add_argument("--batch_size_validation", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--gamma", type=float, default=0.85, help="Learning rate decay factor.")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_train_batches", type=int, default=-1, help="Maximum number of training batches per epoch, -1 if using all training samples")
parser.add_argument("--max_val_batches", type=int, default=-1, help="Maximum number of validation batches, -1 if using all validation samples")
args = parser.parse_args()

train_config.model_name = args.model_name
train_config.train_data_path = args.train_data_path
train_config.validation_data_path = args.validation_data_path
train_config.output_name = args.output_name
train_config.max_words = args.max_words
train_config.batch_size_training = args.batch_size_training
train_config.batch_size_validation = args.batch_size_validation
train_config.gradient_accumulation_steps = args.gradient_accumulation_steps
train_config.num_epochs = args.num_epochs
train_config.gamma = args.gamma
train_config.lr = args.lr
train_config.max_train_batches = args.max_train_batches
train_config.max_val_batches = args.max_val_batches

lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r = 8,
    lora_alpha = 32,
    target_modules = ["q_proj", "v_proj"],
    bias= "none",
    lora_dropout = 0.05,
    inference_mode = False
)

start_time = time.time()

model = load_model(train_config.model_name, quantization=True)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
# tokenizer.add_special_tokens({"pad_token": "<PAD>"})
tokenizer.pad_token_id = tokenizer.eos_token_id

dataset_train = AlpacaDataset(train_config.train_data_path, tokenizer, train_config.max_words)
train_dataloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=train_config.batch_size_training,
    num_workers=1,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    collate_fn=default_data_collator,
)
if train_config.run_validation:
    dataset_validation = AlpacaDataset(train_config.validation_data_path, tokenizer, train_config.max_words)
    validation_dataloader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=train_config.batch_size_validation,
        num_workers=1,
        pin_memory=True,
        collate_fn=default_data_collator,
    )
else:
    validation_dataloader = None

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.lr,
    weight_decay=0.0,
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=train_config.gamma)

results = train(
    model, 
    train_config,
    train_dataloader,
    validation_dataloader,
    optimizer,
    scheduler
)

if train_config.save_model:
    save_peft_model(model, train_config.output_name)
    with open("finetuning_results.txt", "a") as f:
        total_time = str((time.time() - start_time) // 3600)
        f.write(f"{args.train_data_path} {total_time} hours\n")