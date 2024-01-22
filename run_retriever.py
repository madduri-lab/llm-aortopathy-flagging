import os
import copy
import json
import torch
import logging
import argparse
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    CHROMA_SETTINGS
)
def retrieval_DB(device):
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device})
    # This one create Chroma Client and pass it to LangChain
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    return db

def add_context_to_json(source_dir: str, target_dir: str, retriever: Chroma, k: int=1):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)

            with open(source_file, 'r') as file:
                try:
                    data_origin = json.load(file)
                    if not isinstance(data_origin, list):
                        data_origin = [data_origin]
                    data_augment = []
                    for data in data_origin:
                        docs = retriever.similarity_search(data["Q"])
                        if len(docs):
                            for doc in docs[:min(k, len(docs))]:
                                data_cp = copy.copy(data)
                                data_cp['Context'] = doc.page_content
                                data_augment.append(data_cp)
                except Exception as e:
                    print(f"An error occurred during the similarity search: {e}")

            with open(target_file, 'w') as file:
                json.dump(data_augment, file, indent=4)

if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s %(levelname)-4s]: %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description='Document Processing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_context', type=int, default=2, help="number of retrieved contexts for each query.")
    args = parser.parse_args()
    ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

    SOURCE_DIR = f"{ROOT_DIRECTORY}/data/datasets/raw"
    TARGET_DIR = f"{ROOT_DIRECTORY}/data/datasets/rag"
    #Test code below
    db = retrieval_DB(args.device)
    add_context_to_json(SOURCE_DIR, TARGET_DIR, retriever=db, k=args.num_context)
