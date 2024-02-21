"""
This script is used to preprocess the Marfan notes in `marfan_allnotes_022024.csv`by 
(1) retrieving relevant context information from several Marfan-related research papers stored 
in the `data/docs` directory, and augmenting this information to each note in the dataset;
(2) separating the notes into training and testing sets;
(3) and saving the augemented training and testing sets as JSON files.
"""
import csv
import json
import torch
import argparse
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from constants import EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, CHROMA_SETTINGS

argparser = argparse.ArgumentParser()
argparser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
argparser.add_argument("--input", help="Path to the input CSV file", default="./data/datasets/raw/marfan_allnotes_022024.csv")
argparser.add_argument("--output_train", help="Path to the output training dataset", default="./data/datasets/rag/marfan_rag_train.json")
argparser.add_argument("--output_test", help="Path to the output testing dataset", default="./data/datasets/rag/marfan_rag_test.json")
argparser.add_argument("--docs", help="Path to the directory containing the Marfan-related research papers", default="../docs")
argparser.add_argument("--num_context", type=int, default=2, help="number of retrieved contexts for each note.")

args = argparser.parse_args()

# Load the input CSV file
with open(args.input, "r") as f:
    reader = csv.DictReader(f)
    marfan_notes = list(reader)

# Load the Marfan-related research papers
embeddings = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME, 
    model_kwargs={"device": args.device}
)
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS
)

training_set, testing_set = [], []
for note_entry in marfan_notes:
    note, raw_label = note_entry['summary'], note_entry['label']
    set_type, label = raw_label.split("_")
    relevant_docs = db.similarity_search(note)
    if len(relevant_docs):
        for doc in relevant_docs[:min(args.num_context, len(relevant_docs))]:
            if set_type == "training":
                training_set.append({
                    "note_id": note_entry['note_id'],
                    "note_context": doc.page_content,
                    "note_summary": note,
                    "note_label": label,
                })
            else:
                testing_set.append({
                    "note_id": note_entry['note_id'],
                    "note_context": doc.page_content,
                    "note_summary": note,
                    "note_label": label,
                })

with open(args.output_train, "w") as f:
    json.dump(training_set, f, indent=4)

with open(args.output_test, "w") as f:
    json.dump(testing_set, f, indent=4)