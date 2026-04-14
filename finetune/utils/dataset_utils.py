def instruct2alpaca(instruct: dict):
    """Convert an instruction to the Alpaca format."""
    prompt_dict = {
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
    if instruct.get("input", "") == "":
        prompt = prompt_dict["prompt_no_input"].format_map(instruct)
    else:
        prompt = prompt_dict["prompt_input"].format_map(instruct)
    return prompt

def result_process(result):
    """Only obtain the generated part from the result sentence."""
    idx = len(result)
    while result[idx-14:idx+1] != "\n\n### Response:":
        idx -= 1
        if idx == 0:
            print(result)
            break
    return result[idx+1:]

def prediction2supergluelabels(prediction, dataset="CB"):
    """Convert the prediction to the Superglue-format labels."""
    prediction = prediction.lower()
    if dataset == "CB":
        if 'entailment' in prediction: return 'Entailment'
        elif 'contradiction' in prediction: return 'Contradiction'
        else: return 'Neutral'
    elif dataset == "WSC":
        if 'yes' in prediction: return 'Yes'
        else: return 'No'
    elif dataset == "COPA":
        if 'one' in prediction: return 'One'
        else: return 'Two'
    elif dataset == "BoolQ":
        if 'true' in prediction: return 'True'
        else: return 'False'
    elif dataset == "WiC":
        if 'yes' in prediction: return 'Yes'
        else: return 'No'
    elif dataset == "RTE":
        if 'yes' in prediction: return 'Yes'
        else: return 'No'
    elif dataset == "MultiRC":
        if 'yes' in prediction: return 'Yes'
        else: return 'No'
    else:
        raise NotImplementedError