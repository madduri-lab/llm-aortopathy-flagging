import time
import torch
import argparse
from data import *
from train import train
from config import train_config
from utils.model_utils import save_peft_model
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import default_data_collator, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()  
## model
parser.add_argument(
    "--model_name", 
    type=str, 
    default="mistralai/Mistral-7B-v0.1",
    choices=[
        "epfl-llm/meditron-7b",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", 
        "mistralai/Mistral-7B-v0.1", 
    ]
)
parser.add_argument("--lora_size", type=str, default="small", choices=["small", "large"])
parser.add_argument("--output_name", type=str, default="./model/marfan/mistral.pt")

## dataset
parser.add_argument("--dataset_type", type=str, default="RawTextDataset", choices=["RawTextDataset", "AlpacaDataset", "ClinicalNoteDataset", "CleanedTextDataset"])
parser.add_argument("--train_data_path", type=str, default="./data/raw_data/data_cb.json")
parser.add_argument("--validation_data_path", type=str, default="./data/raw_data/data_cb_val.json")
parser.add_argument("--max_tokens", type=int, default=224, help="Maximum number of tokens for the input.")
parser.add_argument("--is_ntp", action="store_true", default=False, help="If the training is for next token prediction")

## training
parser.add_argument("--batch_size_training", type=int, default=4)
parser.add_argument("--batch_size_validation", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.85, help="Learning rate decay factor.")
parser.add_argument("--max_train_batches", type=int, default=-1, help="Maximum number of training batches per epoch, -1 if using all training samples")
parser.add_argument("--max_val_batches", type=int, default=-1, help="Maximum number of validation batches, -1 if using all validation samples")

args = parser.parse_args()

train_config.model_name = args.model_name
train_config.output_name = args.output_name

train_config.train_data_path = args.train_data_path
train_config.validation_data_path = args.validation_data_path
train_config.max_tokens = args.max_tokens

train_config.lr = args.lr
train_config.gamma = args.gamma
train_config.weight_decay = args.weight_decay
train_config.num_epochs = args.num_epochs
train_config.batch_size_training = args.batch_size_training
train_config.batch_size_validation = args.batch_size_validation
train_config.gradient_accumulation_steps = args.gradient_accumulation_steps
train_config.max_train_batches = args.max_train_batches
train_config.max_val_batches = args.max_val_batches

if args.lora_size == "small":
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

start_time = time.time()

model = AutoModelForCausalLM.from_pretrained(
    train_config.model_name,
    return_dict=True,
    load_in_8bit=True,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

dataset_train = eval(args.dataset_type)(
    train_config.train_data_path,
    tokenizer,
    train_config.max_tokens,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=train_config.batch_size_training,
    num_workers=1,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    collate_fn=(
        default_data_collator
        if not args.is_ntp 
        else DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
    ),
)

if train_config.run_validation:
    dataset_validation = eval(args.dataset_type)(
        train_config.validation_data_path,
        tokenizer,
        train_config.max_tokens,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=train_config.batch_size_validation,
        num_workers=1,
        pin_memory=True,
        collate_fn=(
            default_data_collator
            if not args.is_ntp 
            else DataCollatorForLanguageModeling(
                tokenizer=tokenizer, 
                mlm=False
            )
        ),
    )
else:
    validation_dataloader = None

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.lr,
    weight_decay=train_config.weight_decay,
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