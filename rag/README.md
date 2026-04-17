# RAG Pipeline

This folder contains the Retrieval-Augmented Generation (RAG) pipeline used to build a Chroma vector database from Marfan-related literature and augment clinical notes with retrieved context before fine-tuning.

## Pipeline overview

```
ingest.py / ingest_batch.py   →   data/db-new/   →   preprocess.py   →   data/datasets/rag/
  (build vectorstore)             (Chroma DB)       (retrieve context)    (augmented JSON)
```

## Files

### `constants.py`
Shared configuration imported by all other scripts. Key settings:

| Constant | Value | Description |
|---|---|---|
| `SOURCE_DIRECTORY` | `rag/corpus/Marfan_Corpus_Grobid` | Source documents to ingest |
| `PERSIST_DIRECTORY` | `data/db-new` | Where the Chroma vectorstore is saved |
| `EMBEDDING_MODEL_NAME` | `hkunlp/instructor-large` | HuggingFace embedding model |
| `DOCUMENT_MAP` | — | Maps file extensions to LangChain loaders |
| `INGEST_THREADS` | `os.cpu_count()` | Parallelism for document loading |

All paths are resolved relative to the project root (one level above this folder).

---

### `ingest.py`
Loads all documents from `SOURCE_DIRECTORY` into the Chroma vectorstore in one pass. Suitable for smaller document collections.

```bash
python rag/ingest.py --device_type cuda
```

**What it does:**
1. Walks `SOURCE_DIRECTORY` and loads every supported file type in parallel (using `ProcessPoolExecutor`)
2. Splits text into 1000-character chunks (200-character overlap); Python files use a code-aware splitter
3. Embeds all chunks with `hkunlp/instructor-large` and persists them to `PERSIST_DIRECTORY`

---

### `ingest_batch.py`
Same as `ingest.py` but processes documents in batches (default: 10 at a time), embedding and persisting each batch before loading the next. Use this when the full document collection is too large to hold in memory at once.

```bash
python rag/ingest_batch.py --device_type cuda
```

**Difference from `ingest.py`:** Calls `process_and_store_documents()` after each batch rather than accumulating all documents first.

---

### `preprocess.py`
Queries the built Chroma vectorstore to retrieve relevant literature context for each clinical note, then saves augmented train/test splits as JSON files for fine-tuning.

```bash
python rag/preprocess.py \
    --input        ./data/datasets/raw/marfan_allnotes_022024.csv \
    --output_train ./data/datasets/rag/marfan_rag_train.json \
    --output_test  ./data/datasets/rag/marfan_rag_test.json \
    --num_context  2 \
    --device       cuda
```

**What it does:**
1. Reads clinical notes and labels from the input CSV
2. For each note, runs a similarity search against the Chroma DB to retrieve the top `--num_context` passages from Marfan literature
3. Outputs JSON records with fields `note_id`, `note_context`, `note_summary`, and `note_label`, split into training and testing sets based on the label prefix in the CSV

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--input` | `./data/datasets/raw/marfan_allnotes_022024.csv` | Input CSV with `note_id`, `summary`, `label` columns |
| `--output_train` | `./data/datasets/rag/marfan_rag_train.json` | Output path for augmented training set |
| `--output_test` | `./data/datasets/rag/marfan_rag_test.json` | Output path for augmented testing set |
| `--num_context` | `2` | Number of retrieved literature passages per note |
| `--device` | auto-detected | `cuda` or `cpu` |

## Running order

1. Place source documents in `corpus/Marfan_Corpus_Grobid/` (already populated with KnowledgeBase1 and KnowledgeBase2)
2. Run `ingest.py` (or `ingestLarge.py` for large collections) to build the vectorstore
3. Run `preprocess.py` to generate the augmented dataset for fine-tuning

> All scripts should be run from the **project root**, not from inside this folder.
