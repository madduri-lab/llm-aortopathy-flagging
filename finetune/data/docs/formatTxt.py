import os
import re

def format_text(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Remove references like [1], [2], etc., and those with ellipses [9••]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[\d+••\]', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove downloaded-from messages
    text = re.sub(r'Downloaded from by [^\s]+', '', text)

    # Fix spacing issues
    text = re.sub(r'\s+', ' ', text).strip()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

def process_all_files(folder_path):
    formatted_folder_path = os.path.join(folder_path, 'formatted')
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(formatted_folder_path, file_name)
            format_text(file_path, output_path)
            print(f"Processed and saved: {output_path}")

folder_path = '/home/ze/marfan'  # Update this to the actual path of your folder
process_all_files(folder_path)
