import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torch
import argparse
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

def file_log(logentry, log_type="info"):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    if log_type == "info":
        logging.info(logentry)
    elif log_type == "error":
        logging.error(logentry)
    else:
        logging.debug(logentry)

def load_single_document(file_path: str):
    # Loads a single document from a file path
    try:
        file_extension = os.path.splitext(file_path)[1]
        # Test code part
        # if file_extension != '.pdf':
        #     file_log(f'Skipping unsupported file type: {file_path}', "error")
        #     return None

        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
            file_log(f'{file_path} loaded successfully.')
            return loader.load()[0]
        else:
            file_log(f'{file_path} document type is undefined.', "error")
            raise ValueError("Document type is undefined")
    except Exception as ex:
        file_log(f'Error loading {file_path}: {str(ex)}', "error")
        return None


def process_and_store_documents(documents, deviceType):
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": deviceType},
    )
    logging.info("Finish creating the embeddings for the batch")

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    logging.info("Finish creating the database for the batch")

    return


def load_documents_in_batches_and_process(source_dir: str, batch_size: int = 10, device_type = 'cuda' if torch.cuda.is_available() else 'cpu'):
    # Test with PDF only
    # paths = [os.path.join(root, name)
    #          for root, _, files in os.walk(source_dir)
    #          for name in files if os.path.splitext(name)[1] == '.pdf']
    
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print('Importing: ' + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    workers = min(INGEST_THREADS, max(len(paths), 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i in range(0, len(paths), batch_size):
            futures = {executor.submit(load_single_document, path): path for path in paths[i:i + batch_size]}
            batch_docs = []
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    if result:
                        batch_docs.append(result)
                except Exception as exc:
                    logging.error(f'{path} generated an exception: {str(exc)}')
            
            # Process current batch and persist
            process_and_store_documents(batch_docs, device_type)



def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
           file_extension = os.path.splitext(doc.metadata["source"])[1]
           if file_extension == ".py":
               python_docs.append(doc)
           else:
               text_docs.append(doc)
    return text_docs, python_docs


def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    load_documents_in_batches_and_process(SOURCE_DIRECTORY, batch_size=10, device_type=device_type)
    
    logging.info("Finish creating the database")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Document Processing')
    parser.add_argument('--device_type', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 've', 'fpga', 'ort', 'xla', 'lazy', 'vulkan', 'mps', 'meta', 'hpu', 'mtia'],
                        help='Device to run on. (Default is cuda)')
    args = parser.parse_args()

    main(args.device_type)
