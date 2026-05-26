# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook 1 of 5: Setup & Configuration
# MAGIC
# MAGIC This notebook prepares the environment, loads clinical notes, and defines shared
# MAGIC utility functions used across the inference pipeline.
# MAGIC
# MAGIC **Expected input format:** A JSON list of dictionaries with keys:
# MAGIC - `id`: person-level identifier
# MAGIC - `ClinicalNoteKey`: note-level identifier
# MAGIC - `label`: disease diagnosis label (if applicable)
# MAGIC - `tested`: whether individual required genetic testing (`"required"` or `"not required"`)
# MAGIC - `note`: de-identified clinical note text
# MAGIC
# MAGIC All patient notes should be de-identified prior to use (e.g., using
# MAGIC [Philter](https://www.nature.com/articles/s41746-020-0258-y)).
# MAGIC
# MAGIC **Full pipeline (5 notebooks):**
# MAGIC 1. Setup & configuration (this notebook)
# MAGIC 2. Base model inference (`02_base_inference`)
# MAGIC 3. Confidence thresholding & low-confidence ID export (`03_confidence_thresholding`)
# MAGIC 4. RAG inference & merge with base results (`04_rag_inference_and_merge`)
# MAGIC 5. Final evaluation & base vs. final comparison (`05_final_evaluation`)
# MAGIC
# MAGIC Note: This pipeline assumes a genetic aortopathy screening question. If adapting for
# MAGIC a different disease, modify the system prompt and response parsing regex accordingly.

# COMMAND ----------

# DBTITLE 1,Update Databricks Environment
!pip install --upgrade transformers
!pip install -U bitsandbytes
!pip install 'accelerate>=0.26.0'

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Libraries
import sklearn
import mlflow.deployments as mlfd
from datetime import datetime, timezone
import pytz
est = pytz.timezone('US/Eastern')
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import mlflow
import os
import time
import shlex
import shutil
import subprocess
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import re
import json
import math
from typing import Dict, List, Optional
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import random

# COMMAND ----------

# DBTITLE 1,Instantiate Client Connection
import mlflow.deployments as mlfd
from datetime import datetime, timezone
import pytz

client = mlfd.get_deploy_client("databricks")
est = pytz.timezone('US/Eastern')  # Databricks defaults to UTC

# Verify available endpoints
print("Available model endpoints:")
for ep in client.list_endpoints():
    print(f"  - {ep['name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration
# MAGIC Set **all** file paths and model parameters below. These variables are referenced
# MAGIC across all 5 notebooks in this pipeline.

# COMMAND ----------

# DBTITLE 1,*** Set Configuration (EDIT THIS CELL)
# ═══════════════════════════════════════════════════════════════════════════════
# FILE PATHS — Update to match your Databricks volume / workspace
# ═══════════════════════════════════════════════════════════════════════════════

# Input
INPUT_DATA_PATH = "<YOUR_VOLUME_PATH>/input_notes_data.json"

# Notebook 2 outputs (base inference)
STEP1_VALID_OUTPUT_PATH = "<YOUR_VOLUME_PATH>/step1_base_valid_results.json"
STEP1_POORLY_FORMATTED_OUTPUT_PATH = "<YOUR_VOLUME_PATH>/step1_base_poorly_formatted.json"

# Notebook 3 output (low-confidence note IDs)
LOW_CONF_IDS_OUTPUT_PATH = "<YOUR_VOLUME_PATH>/step1_low_confidence_note_ids.csv"

# Notebook 4 inputs/outputs (RAG inference + merge)
RAG_AUGMENTED_INPUT_PATH = "<YOUR_VOLUME_PATH>/rag_augmented_prompts.json"
STEP2_VALID_OUTPUT_PATH = "<YOUR_VOLUME_PATH>/step2_rag_valid_results.json"
STEP2_POORLY_FORMATTED_OUTPUT_PATH = "<YOUR_VOLUME_PATH>/step2_rag_poorly_formatted.json"
MERGED_RESULTS_OUTPUT_PATH = "<YOUR_VOLUME_PATH>/final_merged_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_ENDPOINT = "<YOUR_MODEL_ENDPOINT>"   # e.g., "meta-llama-3-1-8b-instruct"
TEMPERATURE = 0.3                          # Must be identical in Steps 1 & 2
MAX_TOKENS = 300
CONFIDENCE_THRESHOLD = 0.5                 # Notes below this are rerun with RAG

# COMMAND ----------

# DBTITLE 1,Load Notes Data
with open(INPUT_DATA_PATH, "r") as file:
    prompts_data = json.load(file)

print(f"Total notes in dataset: {len(prompts_data)}")
print(f"Unique patients: {len(set(d['id'] for d in prompts_data))}")

# Preview first record (note text truncated for display)
sample = {k: (v[:100] + '...' if k == 'note' and len(str(v)) > 100 else v)
          for k, v in prompts_data[0].items()}
print(f"\nSample record:\n{json.dumps(sample, indent=2)}")

# Option: subset for testing
# prompts_data = random.sample(prompts_data, k=5)

# COMMAND ----------

# DBTITLE 1,Set System Prompt
# Modify this prompt for your specific disease/phenotype of interest.
# The JSON response format {'testing': 'recommended'/'not recommended'} is
# expected by the downstream parsing regex — keep that structure consistent.
system_prompt = """You are a clinical expert on rare genetic diseases, with a specialization in genetic aortopathic conditions such as Marfan syndrome, Loeys-Dietz syndrome, and similar disorders. Your task is to determine if this patient needs genetic testing specifically for aortopathic genetic diseases based on their past and present symptoms and medical history.

Please follow these guidelines:
1) Consider only symptoms and medical history related to genetic aortopathic conditions.
2) If the patient shows signs that suggest an genetic aortopathic disease, recommend testing and provide specific criteria why.
3) If the patient does not show signs specific to genetic aortopathic diseases, state why genetic testing for these conditions is not recommended.

Return your response as a JSON formatted string with 2 parts:
1) testing recommendation {'testing':'recommended'} or {'testing':'not recommended'}
2) your reasoning, focused solely on genetic aortopathic conditions
"""

print("System prompt set.")

# COMMAND ----------

# DBTITLE 1,Define Shared Utility Functions
def validate_response_format(response_content: str) -> Optional[int]:
    """Check if LLM response contains a parseable testing recommendation.

    Returns 1 (recommended), 0 (not recommended), or None (unparseable).
    """
    is_recommended = re.search(
        r'"?testing"?\s*:\s*"?recommended"?', response_content, re.IGNORECASE
    )
    is_not_recommended = re.search(
        r'"?testing"?\s*:\s*"?not recommended"?', response_content, re.IGNORECASE
    )
    if is_recommended:
        return 1
    elif is_not_recommended:
        return 0
    return None


def get_response_with_logprobs(prompt: Dict, client, system_prompt: str,
                                attempt: int) -> Dict:
    """Query model endpoint and extract a log-probability confidence score.

    Extracts P("recommended") or P("not")*P("recommended") from token-level
    logprobs to produce a single scalar confidence for the prediction.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Clinical Note: {prompt['note']}"}
    ]

    response = client.predict(
        endpoint=MODEL_ENDPOINT,
        inputs={
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "logprobs": True
        }
    )

    response_content = response['choices'][0]['message']['content'].strip()
    logprobs_content = response['choices'][0]['logprobs']['content']

    recommended_prob = None
    combined_prob = None
    not_token_logprob = None

    if validate_response_format(response_content) is not None:
        not_flag = False
        for item in logprobs_content:
            if 'recommended' in item['token'] and not_flag:
                combined_prob = math.exp(not_token_logprob + item['logprob'])
                break
            if 'recommended' in item['token']:
                recommended_prob = math.exp(item['logprob'])
                break
            if 'not' in item['token']:
                not_flag = True
                not_token_logprob = item['logprob']

    return {
        "ClinicalNoteKey": prompt['ClinicalNoteKey'],
        "id": prompt["id"],
        "label": prompt["label"],
        "tested": prompt["tested"],
        "response": response_content,
        "attempt": attempt,
        "final_probability": combined_prob if combined_prob is not None else recommended_prob
    }


print("Utility functions defined: validate_response_format, get_response_with_logprobs")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup complete.
# MAGIC
# MAGIC Variables ready for downstream notebooks:
# MAGIC - `prompts_data` — loaded clinical notes
# MAGIC - `system_prompt` — LLM system prompt
# MAGIC - `client` — Databricks model serving client
# MAGIC - `validate_response_format()` / `get_response_with_logprobs()` — shared functions
# MAGIC - All path and parameter constants from the configuration cell
# MAGIC
# MAGIC **Next:** Run `02_base_inference`.
