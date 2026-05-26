# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook 4 of 5: RAG Inference & Merge
# MAGIC
# MAGIC Loads RAG-augmented prompts (generated on HPC), runs inference on the low-confidence
# MAGIC notes using the same model + temperature, then merges RAG results with the
# MAGIC high-confidence base results into a single unified prediction set.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Run `01_setup_and_config` first
# MAGIC - Run `02_base_inference` (produces base valid results)
# MAGIC - Run `03_confidence_thresholding` (exports low-confidence IDs)
# MAGIC - Complete the HPC RAG augmentation step (described at the end of Notebook 3)
# MAGIC
# MAGIC **Outputs:**
# MAGIC - RAG inference valid results JSON
# MAGIC - Merged final results JSON (high-confidence base + RAG rerun)

# COMMAND ----------

# DBTITLE 1,Load RAG Augmented Prompts
with open(RAG_AUGMENTED_INPUT_PATH, "r") as file:
    rag_augmented_data = json.load(file)

print(f"RAG augmented notes loaded: {len(rag_augmented_data)}")

# COMMAND ----------

# DBTITLE 1,Set RAG System Prompt
# The RAG prompt adds "utilize the relevant medical literature provided" to guide
# the model to incorporate the retrieved context appended to each note.
system_prompt_rag = """You are a clinical expert on rare genetic diseases, with a specialization in genetic aortopathic conditions such as Marfan syndrome, Loeys-Dietz syndrome, and similar disorders. Your task is to determine if this patient needs genetic testing specifically for aortopathic genetic diseases based on their past and present symptoms and medical history.

Please follow these guidelines and utilize the relevant medical literature provided:
1) Consider only symptoms and medical history related to genetic aortopathic conditions.
2) If the patient shows signs that suggest an genetic aortopathic disease, recommend testing and provide specific criteria why.
3) If the patient does not show signs specific to genetic aortopathic diseases, state why genetic testing for these conditions is not recommended.

Return your response as a JSON formatted string with 2 parts:
1) testing recommendation {'testing':'recommended'} or {'testing':'not recommended'}
2) your reasoning, focused solely on genetic aortopathic conditions
"""

# Update system_prompt in the augmented data
for entry in rag_augmented_data:
    entry['system_prompt'] = system_prompt_rag

print("RAG system prompt set and applied to all entries.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG Inference
# MAGIC
# MAGIC Same inference loop as the base model (Notebook 2) — up to 3 retries for
# MAGIC malformed responses, logprob extraction for confidence. Temperature and model
# MAGIC must match Step 1 for a fair comparison.

# COMMAND ----------

# DBTITLE 1,Run RAG Inference
rag_valid_results = []
rag_poorly_formatted = []

for prompt in rag_augmented_data:
    print(f"\nProcessing note ID: {prompt['ClinicalNoteKey']}")
    valid_format = False
    final_result = None

    for attempt in range(1, 4):
        result = get_response_with_logprobs(prompt, client, system_prompt_rag, attempt)
        print(f"Attempt: {attempt}")
        print(f"Response: {result['response']}\n")

        if validate_response_format(result['response']) is not None:
            valid_format = True
            final_result = result
            print(f"Valid format achieved on attempt {attempt}")
            print(f"Final probability: {result['final_probability']}")
            break
        else:
            print(f"Invalid format detected on attempt {attempt}")
            if attempt < 3:
                print("Trying again...")

    if valid_format:
        rag_valid_results.append(final_result)
        print(f"Successfully processed note ID: {prompt['ClinicalNoteKey']}")
    else:
        rag_poorly_formatted.append(result)
        print(f"\nFailed for note ID: {prompt['ClinicalNoteKey']} after 3 attempts")

if rag_valid_results:
    with open(STEP2_VALID_OUTPUT_PATH, "w") as file:
        json.dump(rag_valid_results, file, indent=4)
    print(f"\nRAG valid results saved to {STEP2_VALID_OUTPUT_PATH}")

if rag_poorly_formatted:
    with open(STEP2_POORLY_FORMATTED_OUTPUT_PATH, "w") as file:
        json.dump(rag_poorly_formatted, file, indent=4)
    print(f"RAG poorly formatted saved to {STEP2_POORLY_FORMATTED_OUTPUT_PATH}")

print(f"\nRAG Processing Summary:")
print(f"  Successfully formatted: {len(rag_valid_results)}")
print(f"  Poorly formatted: {len(rag_poorly_formatted)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base vs. RAG Probability Comparison
# MAGIC
# MAGIC 1:1 comparison of confidence scores for notes that were rerun. Notes that failed
# MAGIC formatting in RAG keep their base-model prediction. The count of matched notes
# MAGIC may therefore be smaller than the total RAG input.

# COMMAND ----------

# DBTITLE 1,Probability Distribution: Base vs RAG
# Load base results
with open(STEP1_VALID_OUTPUT_PATH, 'r') as f:
    base_valid_results = json.load(f)

# Load RAG results (or use in-memory rag_valid_results)
with open(STEP2_VALID_OUTPUT_PATH, 'r') as f:
    rag_valid_results = json.load(f)

rag_keys = {item['ClinicalNoteKey'] for item in rag_valid_results}
filtered_base_results = [item for item in base_valid_results if item['ClinicalNoteKey'] in rag_keys]

print(f"Original base results: {len(base_valid_results)}")
print(f"RAG results: {len(rag_valid_results)}")
print(f"Matched base results: {len(filtered_base_results)}")

# Build paired comparison
rag_dict = {item['ClinicalNoteKey']: item for item in rag_valid_results}
comparison_df = pd.DataFrame([
    {
        'ClinicalNoteKey': base['ClinicalNoteKey'],
        'base_prob': base['final_probability'],
        'rag_prob': rag_dict[base['ClinicalNoteKey']]['final_probability']
    }
    for base in filtered_base_results
    if base['ClinicalNoteKey'] in rag_dict
])

# Density plot
plt.figure(figsize=(8, 5))
sns.kdeplot(data=comparison_df, x='base_prob', label='Base Model', color='blue', alpha=0.5)
sns.kdeplot(data=comparison_df, x='rag_prob', label='RAG Model', color='red', alpha=0.5)
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distributions: Base vs RAG (Low-Confidence Notes Only)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

difference = comparison_df['rag_prob'] - comparison_df['base_prob']
print("\nProbability Difference (RAG - Base):")
print(f"  Mean:   {difference.mean():.3f}")
print(f"  Median: {difference.median():.3f}")
print(f"  Std:    {difference.std():.3f}")
print(f"  Min:    {difference.min():.3f}")
print(f"  Max:    {difference.max():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge Base + RAG Results
# MAGIC
# MAGIC For each note: if it was rerun with RAG and produced a valid result, use the RAG
# MAGIC prediction. Otherwise, keep the original base prediction.

# COMMAND ----------

# DBTITLE 1,Merge Results
rag_dict = {item['ClinicalNoteKey']: item for item in rag_valid_results}

final_results = []
for base_item in base_valid_results:
    clinical_note_key = base_item['ClinicalNoteKey']
    if clinical_note_key in rag_dict:
        final_results.append(rag_dict[clinical_note_key])
    else:
        final_results.append(base_item)

print(f"Base results: {len(base_valid_results)}")
print(f"RAG results: {len(rag_valid_results)}")
print(f"Final merged results: {len(final_results)}")
print(f"Entries replaced by RAG: {len(rag_valid_results)}")

with open(MERGED_RESULTS_OUTPUT_PATH, "w") as f:
    json.dump(final_results, f, indent=4)
print(f"\nMerged results saved to {MERGED_RESULTS_OUTPUT_PATH}")

# Verification
if rag_valid_results:
    sample_key = rag_valid_results[0]['ClinicalNoteKey']
    base_entry = next((x for x in base_valid_results if x['ClinicalNoteKey'] == sample_key), None)
    rag_entry = next((x for x in rag_valid_results if x['ClinicalNoteKey'] == sample_key), None)
    final_entry = next((x for x in final_results if x['ClinicalNoteKey'] == sample_key), None)

    print(f"\nVerification for sample ClinicalNoteKey {sample_key}:")
    print(f"  Base probability:  {base_entry['final_probability']}")
    print(f"  RAG probability:   {rag_entry['final_probability']}")
    print(f"  Final probability: {final_entry['final_probability']}")
    print(f"  Final matches RAG: {final_entry['final_probability'] == rag_entry['final_probability']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge complete.
# MAGIC
# MAGIC **Next:** Run `05_final_evaluation` for note- and patient-level metrics on the
# MAGIC merged results, and comparison plots against the base model.
