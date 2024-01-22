import json
import os
import logging
import argparse
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    CHROMA_SETTINGS
)
def retrieval_DB(device_type):
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
    # This one create Chroma Client and pass it to LangChain
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )

    # This one is basic one. Just load Chroma database
    # db = Chroma(
    #     persist_directory=PERSIST_DIRECTORY,
    #     embedding_function=embeddings,
    # )

    return db

def add_context_to_json(source_dir, target_dir, retriever):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)

            with open(source_file, 'r') as file:
                try:
                    data = json.load(file)

                    docs = retriever.similarity_search(data["Q"])
                    if len(docs) < 1:
                        print("There is no relavant context for source file: " + str(source_file))
                        continue
                    context = docs[0].page_content
                    data['Context'] = context
                except Exception as e:
                    print(f"An error occurred during the similarity search: {e}")

            with open(target_file, 'w') as file:
                json.dump(data, file, indent=4)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Document Processing')
    parser.add_argument('--device_type', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 've', 'fpga', 'ort', 'xla', 'lazy', 'vulkan', 'mps', 'meta', 'hpu', 'mtia'],
                        help='Device to run on. (Default is cuda)')
    args = parser.parse_args()
    ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

    SOURCE_DIR = f"{ROOT_DIRECTORY}/data/datasets/raw"
    TARGET_DIR = f"{ROOT_DIRECTORY}/data/datasets/rag"
    #Test code below
    db = retrieval_DB(args.device_type)
    add_context_to_json(SOURCE_DIR, TARGET_DIR, db)


    # query = "A-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING"
    # answer = "True"
    # top_k = 1 

    # try:
    #     # docs = db.similarity_search_with_score(query)
    #     docs = db.similarity_search(query)
    #     lgt = len(docs)
    #     if top_k > lgt:
    #         print("The num of existing similar doc is less than top_k")
    #         top_k = lgt
    #     for i in range(top_k):
    #         doc = docs[i]
    #         context = doc.page_content
    #         # print(doc)
    #         # print(doc.page_content)
    #         # print("--------------------")
    # except Exception as e:
    #     print(f"An error occurred during the similarity search: {e}")

