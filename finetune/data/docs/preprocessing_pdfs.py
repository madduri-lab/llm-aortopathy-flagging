import re
import os
import fitz
import pathlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dirname", type=str, default="./SOURCE_DOCUMENTS")
parser.add_argument("--output_dirname", type=str, default="./SOURCE_DOCUMENTS_TXT")
args = parser.parse_args()

def remove_sections(text, start_markers, end_markers):
    for start_marker in start_markers:
        for end_marker in end_markers:
            pattern = rf"{start_marker}.*?{end_marker}"
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    return text

def extract_text_from_pdf(pdf_path, output_txt_path):
    with fitz.open(pdf_path) as doc:
        extracted_text = ""
        for page in doc:
            page_text = page.get_text()
            page_lines = page_text.split('\n')
            filtered_lines = [line for line in page_lines if len(line) > 30 and not re.search(r'\bPage \d+\b', line)]
            page_text = '\n'.join(filtered_lines)
            extracted_text += page_text

        # Define markers for sections to remove
        start_markers = ['\nReferences', '\nBibliography', '\nAuthor information', '\nCitation']
        end_markers = ['\n', '\f', '\r']

        # Remove specified sections
        # extracted_text = remove_sections(extracted_text, start_markers, end_markers)

        with open(output_txt_path, "w", encoding="utf-8") as output_file:
            output_file.write(extracted_text)

def process_all_pdfs_in_directory(input_dirname, output_dirname):
    if not os.path.exists(output_dirname):
        pathlib.Path(output_dirname).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(input_dirname):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dirname, filename)
            output_txt_path = os.path.join(output_dirname, filename.replace('.pdf', '.txt'))
            extract_text_from_pdf(pdf_path, output_txt_path)

# Specify the directory containing the PDF files
process_all_pdfs_in_directory(
    args.input_dirname,
    args.output_dirname,
)