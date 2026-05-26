# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook 3 of 5: Confidence Thresholding
# MAGIC
# MAGIC Analyzes the distribution of log-probability confidence scores from the base model
# MAGIC inference (Notebook 2) to identify low-confidence notes for RAG rerun.
# MAGIC
# MAGIC **Prerequisite:** Run `01_setup_and_config` and `02_base_inference` first.
# MAGIC
# MAGIC **Outputs:**
# MAGIC - Probability distribution plots by classification category (TP/FP/TN/FN)
# MAGIC - Confidence bin breakdown
# MAGIC - Per-person note confidence analysis
# MAGIC - CSV of low-confidence note IDs for RAG augmentation
# MAGIC - HPC instructions for running the RAG step

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction Probability Distributions
# MAGIC
# MAGIC We use the log-probability of "recommended" / "not recommended" tokens as a
# MAGIC proxy for model confidence. This section visualizes how confidence distributes
# MAGIC across correct and incorrect predictions.

# COMMAND ----------

# DBTITLE 1,Probability Distribution by Classification Category
# Requires valid_results_df from Notebook 2
classification_results = valid_results_df.copy()
classification_results['category'] = 'NA'

classification_results.loc[(classification_results['ground_truth'] == 1) & (classification_results['prediction'] == 1), 'category'] = 'TP'
classification_results.loc[(classification_results['ground_truth'] == 0) & (classification_results['prediction'] == 1), 'category'] = 'FP'
classification_results.loc[(classification_results['ground_truth'] == 0) & (classification_results['prediction'] == 0), 'category'] = 'TN'
classification_results.loc[(classification_results['ground_truth'] == 1) & (classification_results['prediction'] == 0), 'category'] = 'FN'

final_df = classification_results[['ClinicalNoteKey', 'category', 'final_probability']]

print("DataFrame with classifications:")
print(final_df.head())

# Box + strip plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='final_probability', data=final_df)
sns.stripplot(x='category', y='final_probability', data=final_df, color='red', alpha=0.3)
plt.title('Distribution of Probabilities by Classification Category')
plt.xlabel('Category')
plt.ylabel('Probability')

print("\nProbability Statistics by Category:")
print(final_df.groupby('category')['final_probability'].describe())
plt.show()

# Per-category histograms
plt.figure(figsize=(12, 8))
for i, category in enumerate(['TP', 'FP', 'TN', 'FN'], 1):
    plt.subplot(2, 2, i)
    category_data = final_df[final_df['category'] == category]
    sns.histplot(data=category_data, x='final_probability', bins=10)
    plt.title(f'{category} Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confidence Bin Analysis
# MAGIC
# MAGIC Breaks down TP/FP/TN/FN counts by 10% probability bins to help select
# MAGIC an appropriate threshold for RAG rerun.

# COMMAND ----------

# DBTITLE 1,Generate Counts by Confidence Bin
plt.figure(figsize=(12, 6))
sns.histplot(data=final_df, x='final_probability', bins=30, color='lightgray', alpha=0.5)

fp_data = final_df[final_df['category'] == 'FP']
plt.scatter(fp_data['final_probability'], [0]*len(fp_data),
           color='blue', label='False Positives', s=100, alpha=0.6)

fn_data = final_df[final_df['category'] == 'FN']
plt.scatter(fn_data['final_probability'], [0]*len(fn_data),
           color='red', label='False Negatives', s=100, alpha=0.6)

plt.title('Distribution of Probabilities with FN and FP Highlighted')
plt.xlabel('Probability')
plt.ylabel('Count')
plt.axvline(final_df['final_probability'].median(), color='black', linestyle='--', label='Median')
plt.axvline(final_df['final_probability'].quantile(0.25), color='green', linestyle='--', label='25th Percentile')
plt.legend()
plt.show()

print("\nOverall Probability Statistics:")
print(f"  Min:  {final_df['final_probability'].min():.3f}")
print(f"  25th: {final_df['final_probability'].quantile(0.25):.3f}")
print(f"  50th: {final_df['final_probability'].median():.3f}")
print(f"  75th: {final_df['final_probability'].quantile(0.75):.3f}")
print(f"  Max:  {final_df['final_probability'].max():.3f}")

print("\nFalse Negative Probability Statistics:")
print(fn_data['final_probability'].describe())
print("\nFalse Positive Probability Statistics:")
print(fp_data['final_probability'].describe())

# Bin breakdown
bins = np.arange(0, 1.1, 0.1)
labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
final_df['prob_bin'] = pd.cut(final_df['final_probability'], bins=bins, labels=labels)

bin_counts = pd.crosstab(final_df['prob_bin'], final_df['category'])
bin_counts.loc['All'] = bin_counts.sum()
bin_counts['Total'] = bin_counts.sum(axis=1)

print("\nCounts by Probability Range:")
print(bin_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Threshold Selection
# MAGIC
# MAGIC In testing, a threshold of 0.5 is sufficient for rerun.
# MAGIC
# MAGIC In production (no ground truth), we look at counts of "Recommend" vs. "Not Recommend"
# MAGIC and dynamically select thresholds: if the model recommends, we accept lower confidence
# MAGIC (false positives are tolerable); if the model does not recommend, we require higher
# MAGIC confidence (more stringent to avoid missing cases).

# COMMAND ----------

# DBTITLE 1,Apply Confidence Threshold
low_confidence = valid_results_df[valid_results_df['final_probability'] < CONFIDENCE_THRESHOLD]
high_confidence = valid_results_df[valid_results_df['final_probability'] >= CONFIDENCE_THRESHOLD]

low_conf_ids = set(low_confidence['id'])
low_conf_clinicalnotekeys = set(low_confidence['ClinicalNoteKey'])
high_conf_clinicalnotekeys = set(high_confidence['ClinicalNoteKey'])

# False Negatives analysis
fn_low_conf_ids = len([id for id in fn_ids if id in low_conf_ids])

print(f"Threshold: {CONFIDENCE_THRESHOLD}")
print(f"\nFalse Negatives with Low Confidence:")
print(f"  Total FN IDs: {len(fn_ids)}")
print(f"  FN IDs below threshold: {fn_low_conf_ids}")
print(f"  Percentage: {(fn_low_conf_ids/len(fn_ids)*100):.2f}%")

# False Positives analysis
fp_low_conf = len([id for id in fp_ids if id in low_conf_ids])

print(f"\nFalse Positives with Low Confidence:")
print(f"  Total FP IDs: {len(fp_ids)}")
print(f"  FP IDs below threshold: {fp_low_conf}")
print(f"  Percentage: {(fp_low_conf/len(fp_ids)*100):.2f}%")

print(f"\nTotal low-confidence notes: {len(low_conf_clinicalnotekeys)}")
print(f"Total high-confidence notes: {len(high_conf_clinicalnotekeys)}")
print(f"Total low-confidence patients: {len(low_conf_ids)}")

print("\nFN IDs with low confidence:", [id for id in fn_ids if id in low_conf_ids])
print("FP IDs with low confidence:", [id for id in fp_ids if id in low_conf_ids])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-Person Note Breakdown
# MAGIC
# MAGIC For each incorrectly classified person, shows how many of their notes fall
# MAGIC above/below the threshold. Useful for deciding whether to adjust the threshold.

# COMMAND ----------

# DBTITLE 1,Per-Person Breakdown of Notes Below and Above Threshold
threshold = CONFIDENCE_THRESHOLD

low_confidence = valid_results_df[valid_results_df['final_probability'] < threshold]
high_confidence = valid_results_df[valid_results_df['final_probability'] >= threshold]

# False Negatives
fn_confidence_distribution = {}
for fn_id in fn_ids:
    id_notes_low = len(low_confidence[low_confidence['id'] == fn_id])
    id_notes_high = len(high_confidence[high_confidence['id'] == fn_id])
    total_notes = id_notes_low + id_notes_high
    if total_notes > 0:
        proportion_low = id_notes_low / total_notes
        proportion_high = id_notes_high / total_notes
    else:
        proportion_low = proportion_high = 0
    fn_confidence_distribution[fn_id] = {
        'total_notes': total_notes,
        f'notes_below_{threshold}': id_notes_low,
        f'notes_above_{threshold}': id_notes_high,
        f'proportion_below_{threshold}': proportion_low,
        f'proportion_above_{threshold}': proportion_high
    }

print(f"Note confidences for each False Negative ID (threshold = {threshold}):")
for fn_id, stats in fn_confidence_distribution.items():
    print(f"\n  ID: {fn_id}")
    print(f"  Total: {stats['total_notes']} | Below: {stats[f'notes_below_{threshold}']} ({stats[f'proportion_below_{threshold}']*100:.1f}%) | Above: {stats[f'notes_above_{threshold}']} ({stats[f'proportion_above_{threshold}']*100:.1f}%)")

fn_summary = pd.DataFrame.from_dict(fn_confidence_distribution, orient='index')
print("\nFN Summary:")
print(fn_summary)

# False Positives
fp_confidence_distribution = {}
for fp_id in fp_ids:
    id_notes_low = len(low_confidence[low_confidence['id'] == fp_id])
    id_notes_high = len(high_confidence[high_confidence['id'] == fp_id])
    total_notes = id_notes_low + id_notes_high
    if total_notes > 0:
        proportion_low = id_notes_low / total_notes
        proportion_high = id_notes_high / total_notes
    else:
        proportion_low = proportion_high = 0
    fp_confidence_distribution[fp_id] = {
        'total_notes': total_notes,
        f'notes_below_{threshold}': id_notes_low,
        f'notes_above_{threshold}': id_notes_high,
        f'proportion_below_{threshold}': proportion_low,
        f'proportion_above_{threshold}': proportion_high
    }

print(f"\nNote confidences for each False Positive ID (threshold = {threshold}):")
for fp_id, stats in fp_confidence_distribution.items():
    print(f"\n  ID: {fp_id}")
    print(f"  Total: {stats['total_notes']} | Below: {stats[f'notes_below_{threshold}']} ({stats[f'proportion_below_{threshold}']*100:.1f}%) | Above: {stats[f'notes_above_{threshold}']} ({stats[f'proportion_above_{threshold}']*100:.1f}%)")

fp_summary = pd.DataFrame.from_dict(fp_confidence_distribution, orient='index')
print("\nFP Summary:")
print(fp_summary)

# Visualization
fn_mean_below = fn_summary[f'proportion_below_{threshold}'].mean() * 100
fn_mean_above = fn_summary[f'proportion_above_{threshold}'].mean() * 100
fp_mean_below = fp_summary[f'proportion_below_{threshold}'].mean() * 100
fp_mean_above = fp_summary[f'proportion_above_{threshold}'].mean() * 100

categories = ['False Negatives', 'False Positives']
below_means = [fn_mean_below, fp_mean_below]
above_means = [fn_mean_above, fp_mean_above]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, below_means, width, label=f'Below {threshold}', color='lightcoral')
rects2 = ax.bar(x + width/2, above_means, width, label=f'Above {threshold}', color='lightgreen')
ax.set_ylabel('Mean Percentage (%)')
ax.set_title(f'Average Distribution of Notes Below/Above {threshold} Confidence')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Low-Confidence Note IDs
# MAGIC
# MAGIC We filter at the **note level** (ClinicalNoteKey), not person level, to minimize
# MAGIC compute. For a more conservative approach, you could instead rerun all notes
# MAGIC for any person with at least one low-confidence note.

# COMMAND ----------

# DBTITLE 1,Export Low-Confidence IDs for RAG
df_prompts = pd.DataFrame(prompts_data)

low_conf_notes = set(low_confidence['ClinicalNoteKey'])
low_conf_notes_only = df_prompts[df_prompts['ClinicalNoteKey'].isin(low_conf_notes)]

print(f"Original dataset size: {len(df_prompts)}")
print(f"Low-confidence notes to rerun: {len(low_conf_notes_only)}")

id_notekey_df = low_conf_notes_only[['id', 'ClinicalNoteKey']]
print(f"\nShape: {id_notekey_df.shape}")
print(id_notekey_df)

id_notekey_df.to_csv(LOW_CONF_IDS_OUTPUT_PATH, index=False)
print(f"\nSaved to {LOW_CONF_IDS_OUTPUT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## HPC Instructions for RAG Augmentation
# MAGIC
# MAGIC The RAG augmentation step runs on an HPC cluster, not Databricks. Summary:
# MAGIC
# MAGIC **1. Transfer files to HPC:**
# MAGIC Transfer the low-confidence note IDs CSV and original input JSON to your HPC working directory.
# MAGIC Use Databricks CLI, direct download + scp, or any file transfer method.
# MAGIC
# MAGIC **2. Subset input data to low-confidence notes only:**
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC cases = pd.read_json('<YOUR_INPUT_NOTES>.json', orient='records')
# MAGIC low_conf_ids = pd.read_csv('<YOUR_LOW_CONF_IDS>.csv')
# MAGIC subset_df = cases[cases['ClinicalNoteKey'].isin(low_conf_ids['ClinicalNoteKey'])]
# MAGIC subset_df.to_json('<YOUR_INPUT_FOR_RAG>.json', orient='records', indent=4)
# MAGIC ```
# MAGIC
# MAGIC **3. Run the RAG generation script** (e.g., `generate-parallel3.py`) with updated
# MAGIC input/output paths. Connect to a GPU node first:
# MAGIC ```bash
# MAGIC screen -S rag_session
# MAGIC # Request GPU resources — adjust for your HPC scheduler (LSF/SLURM/etc.)
# MAGIC bsub -Is -q gpu -gpu "num=2:mode=exclusive_process" -n 2 -R "rusage[mem=12GB]" 'bash'
# MAGIC conda activate medrag
# MAGIC python generate-parallel3.py
# MAGIC ```
# MAGIC
# MAGIC **4. Transfer augmented output back to Databricks:**
# MAGIC ```bash
# MAGIC # Configure Databricks CLI with your workspace URL and personal access token
# MAGIC ./databricks configure
# MAGIC ./databricks fs cp <LOCAL_RAG_OUTPUT>.json dbfs:<YOUR_VOLUME_PATH>/
# MAGIC ```
# MAGIC
# MAGIC **Next:** Once RAG augmented prompts are in your Databricks volume, run `04_rag_inference_and_merge`.
