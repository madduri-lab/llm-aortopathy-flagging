# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook 2 of 5: Base Model Inference
# MAGIC
# MAGIC Runs all clinical notes through the base LLM (no RAG augmentation).
# MAGIC For each note, the model returns a testing recommendation and we extract
# MAGIC token-level log-probabilities as a confidence score.
# MAGIC
# MAGIC Notes that fail JSON format validation are retried up to 3 times.
# MAGIC
# MAGIC **Prerequisite:** Run `01_setup_and_config` first.
# MAGIC
# MAGIC **Outputs:**
# MAGIC - Valid results JSON (predictions + confidence scores)
# MAGIC - Poorly formatted results JSON (for debugging)
# MAGIC - Note-level and patient-level performance metrics (printed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Base Model Inference Loop

# COMMAND ----------

# DBTITLE 1,Run Base Inference
valid_results = []
poorly_formatted = []

for prompt in prompts_data:
    print(f"\nProcessing note ID: {prompt['ClinicalNoteKey']}")
    valid_format = False
    final_result = None

    # Retry up to 3 times on malformed JSON output
    for attempt in range(1, 4):
        result = get_response_with_logprobs(prompt, client, system_prompt, attempt)
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
        valid_results.append(final_result)
        print(f"Successfully processed note ID: {prompt['ClinicalNoteKey']}")
    else:
        poorly_formatted.append(result)
        print(f"\nFailed to achieve valid format for note ID: {prompt['ClinicalNoteKey']} after 3 attempts")
        print("Original note content:")
        print("-" * 80)
        print(prompt['note'])
        print("-" * 80)

# Save outputs
if valid_results:
    with open(STEP1_VALID_OUTPUT_PATH, "w") as file:
        json.dump(valid_results, file, indent=4)
    print(f"\nValid results saved to {STEP1_VALID_OUTPUT_PATH}")

if poorly_formatted:
    with open(STEP1_POORLY_FORMATTED_OUTPUT_PATH, "w") as file:
        json.dump(poorly_formatted, file, indent=4)
    print(f"Poorly formatted results saved to {STEP1_POORLY_FORMATTED_OUTPUT_PATH}")

print(f"\nProcessing Summary:")
print(f"Successfully formatted: {len(valid_results)}")
print(f"Poorly formatted: {len(poorly_formatted)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Note-Level Performance Metrics (Base Model)
# MAGIC
# MAGIC Ground truth is at the person level and extrapolated to each note. Some notes may not
# MAGIC reflect the phenotype, but inherit the person-level label. Modify if you have distinct
# MAGIC note-level ground truth.

# COMMAND ----------

# DBTITLE 1,Compute Note-Level Metrics
# Load results (or use in-memory valid_results from above)
with open(STEP1_VALID_OUTPUT_PATH, 'r') as f:
    inf_og = json.load(f)

data = []
for note in inf_og:
    response_content = note['response']
    is_recommended = re.search(r'"?testing"?\s*:\s*"?recommended"?', response_content, re.IGNORECASE)
    is_not_recommended = re.search(r'"?testing"?\s*:\s*"?not recommended"?', response_content, re.IGNORECASE)

    if is_recommended:
        prediction = 1
    elif is_not_recommended:
        prediction = 0
    else:
        prediction = None

    data.append({
        'ClinicalNoteKey': note['ClinicalNoteKey'],
        'id': note['id'],
        'prediction': prediction,
        'ground_truth': int(note['tested'] == 'required'),
        'final_probability': note['final_probability']
    })

    if prediction is None:
        print(f"Ambiguous Case — ClinicalNoteKey: {note['ClinicalNoteKey']}")

results = pd.DataFrame(data)
results['prediction'] = pd.Series(results['prediction']).astype('Int64')
results['ground_truth'] = results['ground_truth'].astype(int)

# Counts
counts = {
    'recommendations': sum(1 for d in data if d['prediction'] == 1),
    'non_recommendations': sum(1 for d in data if d['prediction'] == 0),
    'required_tests': sum(1 for d in data if d['ground_truth'] == 1),
    'not_required_tests': sum(1 for d in data if d['ground_truth'] == 0)
}
print("\nCounts:")
for key, value in counts.items():
    print(f"  {key.replace('_', ' ').title()}: {value}")

# Confusion matrix
valid_results_df = results.dropna()
true_positives = ((valid_results_df['ground_truth'] == 1) & (valid_results_df['prediction'] == 1)).sum()
false_negatives = ((valid_results_df['ground_truth'] == 1) & (valid_results_df['prediction'] == 0)).sum()
true_negatives = ((valid_results_df['ground_truth'] == 0) & (valid_results_df['prediction'] == 0)).sum()
false_positives = ((valid_results_df['ground_truth'] == 0) & (valid_results_df['prediction'] == 1)).sum()

metrics = {
    'Accuracy': (true_positives + true_negatives) / len(valid_results_df),
    'Sensitivity (Recall)': true_positives / (true_positives + false_negatives),
    'Specificity': true_negatives / (true_negatives + false_positives),
    'Precision (PPV)': true_positives / (true_positives + false_positives),
    'NPV': true_negatives / (true_negatives + false_negatives),
    'F1-score': 2 * true_positives / (2 * true_positives + false_positives + false_negatives),
}
beta = 3
metrics['F3-score'] = (1 + beta**2) * metrics['Precision (PPV)'] * metrics['Sensitivity (Recall)'] / \
    ((beta**2 * metrics['Precision (PPV)']) + metrics['Sensitivity (Recall)'])

print("\nConfusion Matrix:")
print(f"  TN: {true_negatives}  |  FP: {false_positives}")
print(f"  FN: {false_negatives}  |  TP: {true_positives}")

print("\nNote-Level Performance Metrics (Base):")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}")

if 'final_probability' in results.columns:
    print("\nProbability Analysis:")
    print(f"  Mean: {results['final_probability'].mean():.3f}")
    print(f"  Median: {results['final_probability'].median():.3f}")
    print(f"  Std: {results['final_probability'].std():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Patient-Level Performance Metrics (Base Model)
# MAGIC
# MAGIC We aggregate note-level predictions to the person level using **consensus** (majority vote).
# MAGIC In our experiments, the conservative approach (any note = "recommended" triggers the person label)
# MAGIC produced an inflated false positive rate for this disease.

# COMMAND ----------

# DBTITLE 1,Compute Patient-Level Metrics
expt0 = inf_og
results = pd.DataFrame(expt0)
results['ground_truth'] = np.where(results['tested'] == 'required', 1, 0)

dont_recommend_patterns = r"testing\": \"not recommended"
recommend_patterns = r"testing\": \"recommended"

def get_prediction(response):
    if re.search(recommend_patterns, response, re.IGNORECASE):
        return 'required'
    elif re.search(dont_recommend_patterns, response, re.IGNORECASE):
        return 'none'
    else:
        return 'undetermined'

results['prediction'] = results['response'].apply(get_prediction)

grouped = results.groupby('id')
results['recommended_prediction'] = grouped['prediction'].transform(lambda x: (x == 'required').mean())
results['notrecommended_prediction'] = grouped['prediction'].transform(lambda x: (x == 'none').mean())

# Consensus: majority label across all notes for a given person
results['consensus_label'] = (results['recommended_prediction'] >= results['notrecommended_prediction']).astype(int)

summary_df = results.drop(columns=['response', 'prediction']).drop_duplicates(subset=['id']).reset_index(drop=True)

print(f"Unique patients: {summary_df.shape[0]}")
print(f"Consensus recommendations: {summary_df['consensus_label'].sum()}")
print(f"Ground truth positives: {summary_df['ground_truth'].sum()}")

matches = (summary_df['consensus_label'] == summary_df['ground_truth']).sum()
mismatches = (summary_df['consensus_label'] != summary_df['ground_truth']).sum()
accuracy_base = matches / len(summary_df)

print(f"\nMatches: {matches} | Mismatches: {mismatches}")

base_true_positives = ((summary_df['ground_truth'] == 1) & (summary_df['consensus_label'] == 1)).sum()
base_false_positives = ((summary_df['ground_truth'] == 0) & (summary_df['consensus_label'] == 1)).sum()
base_true_negatives = ((summary_df['ground_truth'] == 0) & (summary_df['consensus_label'] == 0)).sum()
base_false_negatives = ((summary_df['ground_truth'] == 1) & (summary_df['consensus_label'] == 0)).sum()

print(f"\nBase Patient-Level Confusion Matrix:")
print(f"  TN: {base_true_negatives}  |  FP: {base_false_positives}")
print(f"  FN: {base_false_negatives}  |  TP: {base_true_positives}")

precision_base = base_true_positives / (base_true_positives + base_false_positives)
recall_base = base_true_positives / (base_true_positives + base_false_negatives)
f1_score_base = 2 * (precision_base * recall_base) / (precision_base + recall_base)
beta_base = 3
f3_score_base = (1 + beta_base**2) * (precision_base * recall_base) / ((beta_base**2 * precision_base) + recall_base)

print(f"\nBase Patient-Level Metrics:")
print(f"  Accuracy:  {accuracy_base:.4f}")
print(f"  Precision: {precision_base:.4f}")
print(f"  Recall:    {recall_base:.4f}")
print(f"  F1 Score:  {f1_score_base:.4f}")
print(f"  F3 Score:  {f3_score_base:.4f}")

# Per-category patient IDs
tp_ids = summary_df[(summary_df['ground_truth'] == 1) & (summary_df['consensus_label'] == 1)]['id']
fp_ids = summary_df[(summary_df['ground_truth'] == 0) & (summary_df['consensus_label'] == 1)]['id']
tn_ids = summary_df[(summary_df['ground_truth'] == 0) & (summary_df['consensus_label'] == 0)]['id']
fn_ids = summary_df[(summary_df['ground_truth'] == 1) & (summary_df['consensus_label'] == 0)]['id']

for category, ids in [('True Positive', tp_ids), ('False Positive', fp_ids),
                     ('True Negative', tn_ids), ('False Negative', fn_ids)]:
    print(f"\n{category} IDs ({len(ids)}):")
    print(list(ids))

base_classification_df = pd.DataFrame({
    'id': list(tp_ids) + list(fp_ids) + list(tn_ids) + list(fn_ids),
    'classification': ['TP']*len(tp_ids) + ['FP']*len(fp_ids) +
                     ['TN']*len(tn_ids) + ['FN']*len(fn_ids)
})
base_classification_df['correct'] = base_classification_df['classification'].map(
    {'TP': 1, 'TN': 1, 'FP': 0, 'FN': 0}
)

print("\nBase correct vs incorrect:")
print(base_classification_df['correct'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Base inference complete.
# MAGIC
# MAGIC **Next:** Run `03_confidence_thresholding` to analyze prediction confidence
# MAGIC and identify low-confidence notes for RAG rerun.
