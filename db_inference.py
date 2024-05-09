import torch
import logging
from transformers import LlamaTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from model_utils import load_model, load_peft_model

def run_inference(model_name, use_lora, lora_name, input_query_path, output_response_path, 
                  device='cuda', quantization=True, do_sample=True, use_cache=True, reproducible=True, 
                  max_new_tokens=1500, length_penalty=1, temp=1):
    # Set up logging
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(output_response_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Set evaluation configuration
    eval_config = EvalConfig(device=device, quantization=quantization, do_sample=do_sample,
                             use_cache=use_cache, reproducible=reproducible,
                             max_new_tokens=max_new_tokens, length_penalty=length_penalty,
                             temperature=temp)

    # Load and configure Lora
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        lora_dropout=0.05,
        inference_mode=False
    )

    # Load model
    model = load_model(model_name, quantization=quantization)
    if use_lora:
        model = get_peft_model(model, lora_config)
        model = load_peft_model(model, lora_name)
    else:
        print("Using the original model")
    model.eval()

    # Load queries
    with open(input_query_path) as file:
        queries = file.readlines()
    queries = [q.strip() for q in queries]

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.bos_token

    # Run inference
    with torch.no_grad():
        for query in queries:
            batch = tokenizer([query], return_tensors="pt")
            batch = {
                k: v.to(eval_config.device)
                for k, v in batch.items()
            }
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
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(prediction)
            logger.info("==========================\n")
