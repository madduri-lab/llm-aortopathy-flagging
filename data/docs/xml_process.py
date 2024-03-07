"""
Process the xml files output by GROBID.
"""
import json
import pathlib
import logging
import argparse
import xmltodict
from typing import List, Dict, Union, Optional, Tuple

# Magic strings
NO_HEADER = "##NO_HEADER##"

parser = argparse.ArgumentParser(description='Convert XML to JSON')
parser.add_argument('--xml_files_dir', type=str, help='Directory containing the input XML files')
parser.add_argument('--json_files_dir', type=str, help='Directory to save the output JSON files')
parser.add_argument('--log_file', type=str, help='Log file path', default='process.log')
parser.add_argument('--error_files_dir', type=str, help='Directory to save the error files', default='error_files')

args = parser.parse_args()

def process_abstract(abstract: Union[Dict, List[Dict]]) -> str:
    """Process the paper abstract."""
    if isinstance(abstract, list):
        return "\n".join(process_abstract(para) for para in abstract)
    else:
        if "head" in abstract:
            return f"{abstract['head']} - {abstract['p']}"
        else:
            return abstract["p"]

def process_paragraph(paragraph: Union[str, Dict, List]) -> str:
    """Process the paragraph(s) in a section."""
    if isinstance(paragraph, str):
        return paragraph
    elif isinstance(paragraph, dict):
        return paragraph["#text"]
    elif isinstance(paragraph, list):
        return "\n".join(process_paragraph(sub_paragraph) for sub_paragraph in paragraph)

def process_section(section: Union[Dict, str]) -> Optional[Tuple[str, str]]:
    """Process a section of the paper. Return the section header and the paragraph(s) in the section."""
    if isinstance(section, str):
        return NO_HEADER, section
    if not "p" in section:
        return None
    if "head" in section:
        head = section["head"]
        if isinstance(head, dict):
            head = head["#text"]
    else:
        head = NO_HEADER
    paragraph = process_paragraph(section["p"])
    return head, paragraph

def parse_xml(xml_file_path: str, json_file_path: str):
    with open(xml_file_path, 'r', encoding='utf-8') as file:
        xml_string = file.read()
    dict_data = xmltodict.parse(xml_string)["TEI"]
    parsed_data = []
    # process the header (title and abstract)
    header = dict_data["teiHeader"]
    try:
        parsed_data.append({
            "Title": header["fileDesc"]["titleStmt"]["title"]["#text"]
        })
    except KeyError:
        pass
    try:
        parsed_data.append({
            "Abstract": process_abstract(header["profileDesc"]["abstract"]["div"])
        })
    except:
        pass
    # process the body
    if dict_data["text"]["body"] is None:
        return
    body = dict_data["text"]["body"]["div"]
    for section in body:
        section = process_section(section)
        if section is not None:
            head, paragraph = section
            parsed_data.append({head: paragraph})
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(parsed_data, json_file, indent=4)

# Set up logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(args.log_file)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

if not os.path.exists(args.json_files_dir):
    pathlib.Path(args.json_files_dir).mkdir(parents=True, exist_ok=True)
xml_files = [f for f in os.listdir(args.xml_files_dir) if f.endswith('.xml')]

for xml_file in xml_files:
    xml_file_path = os.path.join(args.xml_files_dir, xml_file)
    json_file_path = os.path.join(args.json_files_dir, xml_file.replace('.xml', '.json'))
    try:
        parse_xml(xml_file_path, json_file_path)
    except Exception as e:
        logger.info(f"Error processing {xml_file}: {e}")
        if not os.path.exists(args.error_files_dir):
            pathlib.Path(args.error_files_dir).mkdir(parents=True, exist_ok=True)
        error_file_path = os.path.join(args.error_files_dir, xml_file)
        with open(xml_file_path, 'r', encoding='utf-8') as file:
            xml_string = file.read()
        with open(error_file_path, 'w', encoding='utf-8') as file:
            file.write(xml_string)
        
