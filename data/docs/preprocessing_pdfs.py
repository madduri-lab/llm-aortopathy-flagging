import fitz  # PyMuPDF
import re
import os

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
        extracted_text = remove_sections(extracted_text, start_markers, end_markers)

        with open(output_txt_path, "w", encoding="utf-8") as output_file:
            output_file.write(extracted_text)

def process_all_pdfs_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            output_txt_path = os.path.join(directory_path, filename.replace('.pdf', '.txt'))
            extract_text_from_pdf(pdf_path, output_txt_path)

# Specify the directory containing the PDF files
directory_path = "/Users/pankhurisinghal/Desktop/Marfan_LLM_Proj/Webscraping_RAG/preprocessing_pdfs"
process_all_pdfs_in_directory(directory_path)