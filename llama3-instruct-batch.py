import torch
import json
import argparse
import re
import os
import logging as py_logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging
from peft import get_peft_model, LoraConfig, TaskType
from utils.model_utils import load_peft_model

# Configure Python logging
py_logging.basicConfig(filename='error.log', level=py_logging.ERROR)

parser = argparse.ArgumentParser()  
## model
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--use_lora", type=str, default="False", choices=["True", "False"])
parser.add_argument("--lora_name", type=str, default="/home/zeyang/models/llama3_instruct_8b_genrev_aora_raw_large.pt")
parser.add_argument("--bypass", type=str, default="True", choices=["True", "False"])

## data
parser.add_argument("--eval_data_path", type=str, default="/home/zeyang/test_output.json")
parser.add_argument("--shot_data_path", type=str, default="/home/zeyang/shots.json")
parser.add_argument("--use_shots", type=str, default="True", choices=["True", "False"])

## parameters for LLM generation
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--temp", type=float, default=0.6)
parser.add_argument("--topp", type=float, default=0.9)
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum numbers of tokens to generate.") 
parser.add_argument("--batch_size", type=int, default=2, help="Batch size for generation")

args = parser.parse_args()

if "large" not in args.lora_name:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        lora_dropout=0.05,
        inference_mode=False
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

if args.bypass == "True":
    # Bypass "Special tokens have been added in the vocabulary..." warning
    logging.set_verbosity_error()

# The new quant setting method.
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
)    

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    return_dict=True,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=quant_config
)

if args.use_lora == "True":
    model = get_peft_model(model, lora_config)
    model = load_peft_model(model, args.lora_name)
else:
    print("Use the original model")

model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")

input_json = json.load(open(args.eval_data_path))
input_shot = json.load(open(args.shot_data_path))[0]

res = []

def extract_json_from_string(string):
    start = -1
    stack = 0
    results = []
    
    json_regex = re.compile(r'({[^{}]*})')
    matches = json_regex.findall(string)
    
    # Try to find a valid JSON object with the required fields
    for match in matches:
        try:
            json_obj = json.loads(match)
            # Check if the JSON object contains only the desired fields
            if set(json_obj.keys()) == {"testing_recommendation", "reasoning"}:
                results.append(match)
        except json.JSONDecodeError:
            continue
    
    # Earlier return
    if len(results) > 0:
        return results[0]
    # Try to deal unfinished case
    try:
        potential_json = string+"}"
        json_obj = json.loads(potential_json)
        if len(results) == 0 and set(json_obj.keys()) == {"testing_recommendation", "reasoning"}:
            results.append(potential_json)
    except:
        pass

    # Earlier return
    if len(results) > 0:
        return results[0]
    
    # Try to deal with nested one.
    for i, char in enumerate(string):
        if char == '{':
            if stack == 0:
                start = i
            stack += 1
        elif char == '}':
            stack -= 1
            if stack == 0 and start != -1:
                end = i + 1
                potential_json = string[start:end]
                try:
                    json_obj = json.loads(potential_json)
                    # Check if the JSON object contains only the desired fields
                    if set(json_obj.keys()) == {"testing_recommendation", "reasoning"}:
                        results.append(potential_json)
                except json.JSONDecodeError:
                    continue
    # Return the first valid JSON object found or None if none found
    return results[0] if results else None

def create_chat_input(note, input_shot, use_shots):
    if use_shots == "True":
        chat = [
            {"role": "system", "content": note['system_prompt']},
            {"role": "user", "content": input_shot['prompt-shot1']},
            {"role": "assistant", "content": input_shot['output-shot1']},
            {"role": "user", "content": input_shot['prompt-shot2']},
            {"role": "assistant", "content": input_shot['output-shot2']},
            {"role": "user", "content": input_shot['prompt-shot3']},
            {"role": "assistant", "content": input_shot['output-shot3']},
            {"role": "user", "content": input_shot['prompt-shot4']},
            {"role": "assistant", "content": input_shot['output-shot4']},
            {"role": "user", "content": note['user_prompt']}
        ]
    else:
        chat = [
            {"role": "system", "content": note['system_prompt']},
            {"role": "user", "content": note['user_prompt']}
        ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # return chat

# Ensure that the tokenizer has a pad token
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

with torch.no_grad():
    for i in range(0, len(input_json), args.batch_size):
        batch_notes = input_json[i:i+args.batch_size]
        batch_chats = [create_chat_input(note, input_shot, args.use_shots) for note in batch_notes]
        
        batch_encodings = tokenizer(
            batch_chats,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        ).to(args.device)

        input_ids = batch_encodings["input_ids"]
        attention_mask = batch_encodings["attention_mask"]

        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            if eos_token_id is None:
                eos_token_id = tokenizer.pad_token_id
        
        try:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_token_id,
                do_sample=True,
                temperature=args.temp,
                top_p=args.topp,
                pad_token_id=tokenizer.pad_token_id
            )
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory. Reducing batch size or max_new_tokens might help.")
            torch.cuda.empty_cache()
            continue

        for j, note in enumerate(batch_notes):
            response = outputs[j][input_ids.shape[1]:] 
            prediction = tokenizer.decode(response, skip_special_tokens=True)
            # print(f"Generated response for note ID {note['id']}: {prediction}")
            try:
                json_string = extract_json_from_string(prediction)
                if json_string:
                    prediction_json = json.loads(json_string)
                    prediction_json['id'] = note['id']
                    res.append(prediction_json)
                else:
                    raise json.JSONDecodeError("No JSON found", prediction, 0)
            except json.JSONDecodeError:
                error_dir = './wrong_format'
                os.makedirs(error_dir, exist_ok=True)
                error_file = os.path.join(error_dir, f"{note['id']}.txt")
                with open(error_file, 'w') as f:
                    f.write(prediction)
                py_logging.error(f"JSONDecodeError for note ID {note['id']}: {str(json.JSONDecodeError)}")
            except Exception as e:
                error_dir = './wrong_format'
                os.makedirs(error_dir, exist_ok=True)
                error_file = os.path.join(error_dir, f"{note['id']}.txt")
                with open(error_file, 'w') as f:
                    f.write(prediction)
                py_logging.error(f"Exception for note ID {note['id']}: {str(e)}")
        
        torch.cuda.empty_cache()  # manually empty cache

with open("output.json", 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)

print("Processing complete.")

