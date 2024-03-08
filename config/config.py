from dataclasses import dataclass

@dataclass
class train_config:
    model_name = "/projects/bbke/zl52/llama2/llama/models_hf/13B"
    train_data_path = "./data/raw_data/data_cb.json"
    validation_data_path = "./data/raw_data/data_cb_val.json"
    max_tokens = 224
    batch_size_training = 4
    batch_size_validation = 4
    lr = 1e-4
    gradient_accumulation_steps = 1
    num_epochs: int = 3
    gamma: float = 0.85
    weight_decay: float = 0.01
    output_name: str = "./model/cb/lora.pt"
    save_model: bool = True
    device: str = "cuda"
    run_validation: bool = False
    max_train_batches: int = -1
    max_val_batches: int = -1

@dataclass
class eval_config:
    model_name: str = "/projects/bbke/zl52/llama2/llama/models_hf/13B"
    peft_model_path: str = "./model/cb/lora_13B.pt"
    dataset: str = "CB"
    validation_data_path: str = "./data/raw_data/data_cb_val.json"
    batchsize: int = 4
    quantization: bool = True
    do_sample: bool = True
    min_length = None
    use_cache: bool = True
    top_p: float = 1.0
    temperature: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    reproducible: bool = True
    seed: int = 42 # For reproducibility
    device: str = "cuda"
    max_new_tokens: int = 3