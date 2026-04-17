import os
import time
import torch
import argparse
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import (
    FeatureAblation,
    LayerIntegratedGradients,
    LayerGradientXActivation,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
)

def load_peft_model(model, peft_model_path):
    """Load the saved peft model into the based model."""
    peft_state_dict = torch.load(peft_model_path, map_location="cuda")
    model.load_state_dict(peft_state_dict, strict=False)
    return model

parser = argparse.ArgumentParser()
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
parser.add_argument(
    "--method",
    type=str,
    default="perturbation",
    choices=["perturbation", "ig", "gxa"],
    help=(
        "Attribution method: "
        "'perturbation' = FeatureAblation (slowest, model-agnostic); "
        "'ig' = Integrated Gradients; "
        "'gxa' = Gradient x Activation (fastest)"
    ),
)
parser.add_argument(
    "--n_steps",
    type=int,
    default=50,
    help="Number of approximation steps for Integrated Gradients (ignored for other methods)",
)
parser.add_argument("--note", type=str, default=None, help="Path to a plain-text clinical note file")
parser.add_argument("--target", type=str, default=None, help="Path to a plain-text file containing the target output string")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save sq_attr_raw.npy and token_attr_raw.npy")
args = parser.parse_args()

# LoRA config
if "large" not in args.lora_name:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        lora_dropout=0.05,
    )
else:
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none",
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )

# Gradient-based methods require float16; perturbation works with 8-bit quantization
if args.method == "perturbation":
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        return_dict=True,
        load_in_8bit=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

if args.use_lora == "True":
    model = get_peft_model(model, lora_config)
    model = load_peft_model(model, args.lora_name)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
tokenizer.pad_token = tokenizer.bos_token

# Load note and target from files if provided, otherwise use built-in defaults
if args.note is not None:
    with open(args.note, "r") as f:
        eval_prompt = f.read()
else:
    eval_prompt = (
        "**Instruction**: You are a clinical expert on rare genetic diseases. "
        "Does this patient need genetic testing for a potential undiagnosed rare genetic disease or syndrome "
        "based on their past and present symptoms and medical history? If yes, provide specific criteria why, "
        "otherwise state why the patient does not require genetic testing. Return your response as a JSON "
        "formatted string with two parts 1) testing recommendation ('recommended' or 'not recommended') "
        "and 2) your reasoning.\n\n**Patient's Medical Progress Note**: <paste note here>\n\n**Evaluation**: "
    )

if args.target is not None:
    with open(args.target, "r") as f:
        target = f.read()
else:
    target = (
        '{\n'
        '    "designation": "recommended",\n'
        '    "reasoning": "Given the patient\'s complex medical history and family background, '
        'genetic testing for aortopathy-related diseases or syndromes is strongly recommended."\n'
        '}'
    )

# Build attribution
inp = TextTokenInput(
    eval_prompt,
    tokenizer,
    skip_tokens=[1],  # skip <s> BOS token
)

start = time.time()

if args.method == "perturbation":
    fa = FeatureAblation(model)
    llm_attr = LLMAttribution(fa, tokenizer)
    attr_res = llm_attr.attribute(inp, target=target)

elif args.method == "ig":
    lig = LayerIntegratedGradients(model, model.model.embed_tokens)
    llm_attr = LLMGradientAttribution(lig, tokenizer)
    attr_res = llm_attr.attribute(inp, target=target, n_steps=args.n_steps)

elif args.method == "gxa":
    lgxa = LayerGradientXActivation(model, model.model.embed_tokens)
    llm_attr = LLMGradientAttribution(lgxa, tokenizer)
    attr_res = llm_attr.attribute(inp, target=target)

print(f"[{args.method}] seq_attr shape:   {attr_res.seq_attr.shape}")
print(f"[{args.method}] token_attr shape: {attr_res.token_attr.shape}")
print(f"[{args.method}] elapsed: {time.time() - start:.1f}s")

os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, "sq_attr_raw.npy"), attr_res.seq_attr.cpu().numpy())
np.save(os.path.join(args.output_dir, "token_attr_raw.npy"), attr_res.token_attr.cpu().numpy())
