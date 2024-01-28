# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html
import copy
import json
import torch
from torch.utils.data import Dataset

class AlpacaDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_words=224):
        self.instructions = json.load(open(data_path))
        self.max_words = max_words
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
        padding = self.max_words - labeled_prompt.shape[0]
        if padding > 0:
            labeled_prompt = torch.cat((labeled_prompt, torch.zeros(padding, dtype=torch.int64)-1)) # TODO: Check left or right padding
        elif padding < 0:
            labeled_prompt = labeled_prompt[:self.max_words]
        
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