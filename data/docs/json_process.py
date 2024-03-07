"""
Process the json files output by xml_process.py into txt files.
"""
import os
import json
import pathlib
import argparse

# Magic strings
NO_HEADER = "##NO_HEADER##"

parser = argparse.ArgumentParser(description='Convert JSON to TXT')
parser.add_argument('--json_files_dir', type=str, help='Directory containing the input JSON files')
parser.add_argument('--txt_files_dir', type=str, help='Directory to save the output TXT files')
parser.add_argument('--all_txt_file', type=str, help='Path to save the combined TXT file', default='corpus.txt')

args = parser.parse_args()

if not os.path.exists(args.txt_files_dir):
    pathlib.Path(args.txt_files_dir).mkdir(parents=True, exist_ok=True)

all_texts = []
all_json_files = [file for file in os.listdir(args.json_files_dir) if file.endswith('.json')]
for json_file in all_json_files:
    file_texts = []
    with open(os.path.join(args.json_files_dir, json_file), 'r', encoding='utf-8') as file:
        data = json.load(file)
    for section in data:
        for key, value in section.items():
            if key == NO_HEADER:
                all_texts.append(value)
                file_texts.append(value)
            else:
                all_texts.append(f'{key}: {value}')
                file_texts.append(f'{key}: {value}')
    with open(os.path.join(args.txt_files_dir, json_file.replace('.json', '.txt')), 'w', encoding='utf-8') as file:
        file.write("\n".join(file_texts))

with open(args.all_txt_file, 'w', encoding='utf-8') as file:
    file.write("\n".join(all_texts))