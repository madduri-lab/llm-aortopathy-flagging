# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook 5 of 5: Final Evaluation
# MAGIC
# MAGIC Computes note-level and patient-level performance metrics on the merged
# MAGIC (base + RAG) results, and generates comparison plots against the base model.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Run `01_setup_and_config` (libraries, config, utilities)
# MAGIC - Run `02_base_inference` (base metrics variables: `accuracy_base`, `precision_base`, etc.)
# MAGIC - Run `04_rag_inference_and_merge` (merged results file)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Note-Level Metrics (Final Model)

# COMMAND ----------

# DBTITLE 1,Final Note-Level Metrics
with open(MERGED_RESULTS_OUTPUT_PATH, 'r') as f:
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

counts = {
    'recommendations': sum(1 for d in data if d['prediction'] == 1),
    'non_recommendations': sum(1 for d in data if d['prediction'] == 0),
    'required_tests': sum(1 for d in data if d['ground_truth'] == 1),
    'not_required_tests': sum(1 for d in data if d['ground_truth'] == 0)
}
print("\nCounts:")
for key, value in counts.items():
    print(f"  {key.replace('_', ' ').title()}: {value}")

valid_results = results.dropna()
true_positives = ((valid_results['ground_truth'] == 1) & (valid_results['prediction'] == 1)).sum()
false_negatives = ((valid_results['ground_truth'] == 1) & (valid_results['prediction'] == 0)).sum()
true_negatives = ((valid_results['ground_truth'] == 0) & (valid_results['prediction'] == 0)).sum()
false_positives = ((valid_results['ground_truth'] == 0) & (valid_results['prediction'] == 1)).sum()

metrics = {
    'Accuracy': (true_positives + true_negatives) / len(valid_results),
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

print("\nFinal Note-Level Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}")

correctly_formatted_responses_ids = results[results['prediction'].notna()]['ClinicalNoteKey'].tolist()
correctly_formatted_responses_rows = [note for note in inf_og if note['ClinicalNoteKey'] in correctly_formatted_responses_ids]
print(f"\nProperly formatted responses: {len(correctly_formatted_responses_rows)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Patient-Level Metrics (Final Model)

# COMMAND ----------

# DBTITLE 1,Final Patient-Level Metrics
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

results['consensus_label'] = (results['recommended_prediction'] >= results['notrecommended_prediction']).astype(int)

summary_df = results.drop(columns=['response', 'prediction']).drop_duplicates(subset=['id']).reset_index(drop=True)

print(f"Unique patients: {summary_df.shape[0]}")
print(f"Consensus recommendations: {summary_df['consensus_label'].sum()}")
print(f"Ground truth positives: {summary_df['ground_truth'].sum()}")

matches = (summary_df['consensus_label'] == summary_df['ground_truth']).sum()
mismatches = (summary_df['consensus_label'] != summary_df['ground_truth']).sum()
accuracy_final = matches / len(summary_df)

print(f"\nMatches: {matches} | Mismatches: {mismatches}")

final_true_positives = ((summary_df['ground_truth'] == 1) & (summary_df['consensus_label'] == 1)).sum()
final_false_positives = ((summary_df['ground_truth'] == 0) & (summary_df['consensus_label'] == 1)).sum()
final_true_negatives = ((summary_df['ground_truth'] == 0) & (summary_df['consensus_label'] == 0)).sum()
final_false_negatives = ((summary_df['ground_truth'] == 1) & (summary_df['consensus_label'] == 0)).sum()

print(f"\nFinal Patient-Level Confusion Matrix:")
print(f"  TN: {final_true_negatives}  |  FP: {final_false_positives}")
print(f"  FN: {final_false_negatives}  |  TP: {final_true_positives}")

precision_final = final_true_positives / (final_true_positives + final_false_positives)
recall_final = final_true_positives / (final_true_positives + final_false_negatives)
f1_score_final = 2 * (precision_final * recall_final) / (precision_final + recall_final)
beta_final = 3
f3_score_final = (1 + beta_final**2) * (precision_final * recall_final) / ((beta_final**2 * precision_final) + recall_final)

print(f"\nFinal Patient-Level Metrics:")
print(f"  Accuracy:  {accuracy_final:.4f}")
print(f"  Precision: {precision_final:.4f}")
print(f"  Recall:    {recall_final:.4f}")
print(f"  F1 Score:  {f1_score_final:.4f}")
print(f"  F3 Score:  {f3_score_final:.4f}")

tp_ids = summary_df[(summary_df['ground_truth'] == 1) & (summary_df['consensus_label'] == 1)]['id']
fp_ids = summary_df[(summary_df['ground_truth'] == 0) & (summary_df['consensus_label'] == 1)]['id']
tn_ids = summary_df[(summary_df['ground_truth'] == 0) & (summary_df['consensus_label'] == 0)]['id']
fn_ids = summary_df[(summary_df['ground_truth'] == 1) & (summary_df['consensus_label'] == 0)]['id']

for category, ids in [('True Positive', tp_ids), ('False Positive', fp_ids),
                     ('True Negative', tn_ids), ('False Negative', fn_ids)]:
    print(f"\n{category} IDs ({len(ids)}):")
    print(list(ids))

final_classification_df = pd.DataFrame({
    'id': list(tp_ids) + list(fp_ids) + list(tn_ids) + list(fn_ids),
    'classification': ['TP']*len(tp_ids) + ['FP']*len(fp_ids) +
                     ['TN']*len(tn_ids) + ['FN']*len(fn_ids)
})
final_classification_df['correct'] = final_classification_df['classification'].map(
    {'TP': 1, 'TN': 1, 'FP': 0, 'FN': 0}
)

print("\nFinal correct vs incorrect:")
print(final_classification_df['correct'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base vs. Final Model Comparison
# MAGIC
# MAGIC Side-by-side comparison of all patient-level metrics and confusion matrices.
# MAGIC Requires `accuracy_base`, `precision_base`, `recall_base`, `f1_score_base`,
# MAGIC `f3_score_base`, and `base_classification_df` from Notebook 2.

# COMMAND ----------

# DBTITLE 1,Print Side-by-Side Metrics
print("=" * 60)
print("PATIENT-LEVEL PERFORMANCE COMPARISON")
print("=" * 60)

print(f"\n{'Metric':<20} {'Base':>10} {'Final':>10} {'Change':>10}")
print("-" * 52)
for name, base_val, final_val in [
    ('Accuracy',  accuracy_base,  accuracy_final),
    ('Precision', precision_base, precision_final),
    ('Recall',    recall_base,    recall_final),
    ('F1 Score',  f1_score_base,  f1_score_final),
    ('F3 Score',  f3_score_base,  f3_score_final),
]:
    pct = ((final_val - base_val) / base_val) * 100 if base_val != 0 else 0
    print(f"{name:<20} {base_val:>10.4f} {final_val:>10.4f} {pct:>+9.2f}%")

print(f"\n{'Confusion Matrix':<20} {'Base':>10} {'Final':>10}")
print("-" * 42)
print(f"{'True Positives':<20} {base_true_positives:>10} {final_true_positives:>10}")
print(f"{'False Positives':<20} {base_false_positives:>10} {final_false_positives:>10}")
print(f"{'True Negatives':<20} {base_true_negatives:>10} {final_true_negatives:>10}")
print(f"{'False Negatives':<20} {base_false_negatives:>10} {final_false_negatives:>10}")

print(f"\n{'Correct/Incorrect':<20} {'Base':>10} {'Final':>10}")
print("-" * 42)
print(f"{'Correct':<20} {base_classification_df['correct'].value_counts()[1]:>10} {final_classification_df['correct'].value_counts()[1]:>10}")
print(f"{'Incorrect':<20} {base_classification_df['correct'].value_counts()[0]:>10} {final_classification_df['correct'].value_counts()[0]:>10}")

# COMMAND ----------

# DBTITLE 1,Metrics Bar Chart
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'F3 Score'],
    'Base': [accuracy_base, precision_base, recall_base, f1_score_base, f3_score_base],
    'Final': [accuracy_final, precision_final, recall_final, f1_score_final, f3_score_final]
})

plt.figure(figsize=(10, 6))
x = np.arange(len(metrics_df['Metric']))
width = 0.35

plt.bar(x - width/2, metrics_df['Base'], width, label='Base', color='lightblue')
plt.bar(x + width/2, metrics_df['Final'], width, label='Final (Base + RAG)', color='lightgreen')
plt.ylabel('Score')
plt.title('Patient-Level Performance Metrics: Base vs Final')
plt.xticks(x, metrics_df['Metric'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Confusion Matrix Heatmaps
base_cm = np.array([[base_true_negatives, base_false_positives],
                    [base_false_negatives, base_true_positives]])
final_cm = np.array([[final_true_negatives, final_false_positives],
                     [final_false_negatives, final_true_positives]])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(base_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Base Model')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_xticklabels(['Negative', 'Positive'])
ax1.set_yticklabels(['Negative', 'Positive'])

sns.heatmap(final_cm, annot=True, fmt='d', cmap='Greens', ax=ax2)
ax2.set_title('Final Model (Base + RAG)')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_xticklabels(['Negative', 'Positive'])
ax2.set_yticklabels(['Negative', 'Positive'])

plt.suptitle('Patient-Level Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Correct vs Incorrect Predictions
plt.figure(figsize=(8, 6))
correct_counts = pd.DataFrame({
    'Model': ['Base', 'Final'],
    'Correct': [base_classification_df['correct'].value_counts()[1],
                final_classification_df['correct'].value_counts()[1]],
    'Incorrect': [base_classification_df['correct'].value_counts()[0],
                 final_classification_df['correct'].value_counts()[0]]
})

correct_counts.plot(x='Model', y=['Correct', 'Incorrect'], kind='bar')
plt.title('Correct vs Incorrect Patient-Level Predictions')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.legend(title='Prediction Type')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Percentage Improvements
print("\nPercentage Improvements (Final vs Base):")
for metric in metrics_df['Metric']:
    base_val = metrics_df.loc[metrics_df['Metric'] == metric, 'Base'].values[0]
    final_val = metrics_df.loc[metrics_df['Metric'] == metric, 'Final'].values[0]
    pct_change = ((final_val - base_val) / base_val) * 100 if base_val != 0 else 0
    print(f"  {metric}: {pct_change:+.2f}%")
