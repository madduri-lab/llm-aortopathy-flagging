import os
import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a clinical expert on rare genetic diseases, with a specialization in genetic aortopathic conditions such as Marfan syndrome, Loeys-Dietz syndrome, and similar disorders. Your task is to determine if this patient needs genetic testing specifically for aortopathic genetic diseases based on their past and present symptoms and medical history.

Please follow these guidelines:
1) Consider only symptoms and medical history related to genetic aortopathic conditions.
2) If the patient shows signs that suggest an genetic aortopathic disease, recommend testing and provide specific criteria why.
3) If the patient does not show signs specific to genetic aortopathic diseases, state why genetic testing for these conditions is not recommended.

Return your response as a JSON formatted string with 2 parts:
1) testing recommendation {'testing':'recommended'} or {'testing':'not recommended'}
2) your reasoning, focused solely on genetic aortopathic conditions<|eot_id|><|start_header_id|>user<|end_header_id|>

Clinical Note:"""

def edit_distance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # Deletion
                    dp[i][j - 1],    # Insertion
                    dp[i - 1][j - 1] # Substitution
                )

    return dp[m][n]

def word_extraction(text_list):
    word_list = []
    word_index = []
    prev_space = False
    for i, word in enumerate(text_list):
        if word == ' ':
            prev_space = True
            continue
        if word.startswith(' '):
            word_list.append(word.strip())
            word_index.append([i])
        else:
            if prev_space:
                word_list.append(word)
                word_index.append([i])
            else:
                word_list[-1] += word
                word_index[-1].append(i)
        prev_space = False
    return word_list, word_index


argparser = argparse.ArgumentParser()
argparser.add_argument('--temperature', type=float, default=15)
argparser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
argparser.add_argument('--note_path', type=str, default='./notes')
argparser.add_argument('--attr_path', type=str, default='./attrs_gxa')
argparser.add_argument('--output_path', type=str, default='./visualizations_nsp_final_gxa')
argparser.add_argument('--file_name', default=None)
args = argparser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

system_prompt_token = tokenizer.encode(system_prompt, add_special_tokens=False)
system_prompt_token_len = len(system_prompt_token)

# Read all notes from the note path
if args.file_name is not None:
    notes = [f'{args.file_name}.txt']
else:
    notes = [note for note in os.listdir(args.note_path) if note.endswith('.txt')]
notes_name = [note.split('.')[0] for note in notes]

for i in range(len(notes)):
    with open(f'{args.note_path}/{notes[i]}', 'r') as f:
        eval_prompt = f.read()

    eval_prompt_token = tokenizer.encode(eval_prompt, add_special_tokens=False)
    eval_prompt_tokens = tokenizer.convert_ids_to_tokens(eval_prompt_token, skip_special_tokens=False)
    eval_prompt_tokens = eval_prompt_tokens[system_prompt_token_len:][:-1]

    sq_attr = np.load(f'{args.attr_path}/{notes_name[i]}_sq_attr.npy')[1:][system_prompt_token_len:][:-1]
    sq_token_processed = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in eval_prompt_tokens]
    word_list, word_index = word_extraction(sq_token_processed)

    with open(f"keywords/{notes[i]}", "r") as file:
        content = file.read().strip()

    parsed_list = ast.literal_eval(content)
    filtered_index = []
    for medical_term in parsed_list:
        medical_words = medical_term.split()
        for k in range(len(word_list)):
            found = False
            tentative_index = []
            for j, medical_word in enumerate(medical_words):
                if edit_distance(medical_word, word_list[k+j]) <= 1:
                    tentative_index += word_index[k+j]
                    found = True
                else:
                    found = False
                    break
            if found:
                filtered_index += tentative_index

    sq_attr = [np.exp(attr / args.temperature) for attr in sq_attr]
    # Zero out tokens not matched to any medical keyword
    for j in range(len(sq_attr)):
        if j not in filtered_index:
            sq_attr[j] = 0

    max_attr = max(sq_attr)
    sq_attr = [attr / max_attr for attr in sq_attr]
    sq_colors = [plt.cm.Blues(value) for value in sq_attr]

    sq_html_output = ''.join(f'<span style="font-family: Aptos, sans-serif;font-size: 20px; background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]});">{token}</span>' for token, color in zip(sq_token_processed, sq_colors))
    sq_html_output = '<p>' + sq_html_output + '</p>'

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(f'{args.output_path}/{notes_name[i]}.html', 'w') as f:
        f.write(sq_html_output)
