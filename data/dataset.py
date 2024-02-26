import copy
import json
import torch
from torch.utils.data import Dataset

class RawTextDataset(Dataset):
    """
    Dataset for fine-tuning causal language models using raw texts stored in a `.txt` file.
    """
    def __init__(
        self,
        data_path,
        tokenizer,
        max_tokens=4096
    ):
        """
        :param: `data_path` Path to the raw .txt file
        :param: `tokenizer` Tokenizer for the model to be finetuned
        :param: `max_tokens` Number of tokens for each text chunk after preprocess
        """
        with open(data_path, "r") as file:
            texts = [line.strip() for line in file]
        tokens = tokenizer(texts)
        concatenated_tokens = {
            k: sum(tokens[k], [])
            for k in tokens.keys()
        }
        total_length = len(concatenated_tokens[list(concatenated_tokens.keys())[0]])
        if total_length >= max_tokens:
            total_length = (total_length // max_tokens) * max_tokens
        self.datasets = {
            k: [
                t[i : i + max_tokens] for i in range(0, total_length, max_tokens)
            ] for k, t in concatenated_tokens.items()
        }
        self.datasets["labels"] = self.datasets["input_ids"].copy()

    def __len__(self):
        return len(self.datasets["labels"])
    
    def __getitem__(self, index):
        return {
            k: self.datasets[k][index]
            for k in self.datasets.keys()
        }

class AlpacaDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_tokens=224):
        self.instructions = json.load(open(data_path))
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.prompt_dict = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        instruct = self.instructions[index]
        if instruct.get("input", "") == "":
            prompt = self.prompt_dict["prompt_no_input"].format_map(instruct)
        else:
            prompt = self.prompt_dict["prompt_input"].format_map(instruct)
        labeled_prompt = prompt + instruct["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        labeled_prompt = self.tokenizer.encode(labeled_prompt)
        labeled_prompt.append(self.tokenizer.eos_token_id)
        labeled_prompt = torch.tensor(
            labeled_prompt, dtype=torch.int64
        )
        padding = self.max_tokens - labeled_prompt.shape[0]
        if padding > 0:
            labeled_prompt = torch.cat((labeled_prompt, torch.zeros(padding, dtype=torch.int64)-1)) # TODO: Check left or right padding
        elif padding < 0:
            labeled_prompt = labeled_prompt[:self.max_tokens]
        
        labels = copy.deepcopy(labeled_prompt)
        labels[:len(prompt)] = -1
        labeled_prompt_mask = labeled_prompt.ge(0)
        labels_mask = labels.ge(0)
        labeled_prompt[~labeled_prompt_mask] = 0
        labels[~labels_mask] = IGNORE_INDEX
        labeled_prompt_mask = labeled_prompt_mask.float()
        return {
            "input_ids": labeled_prompt,
            "labels": labels,
            "attention_mask": labeled_prompt_mask
        }