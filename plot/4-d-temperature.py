import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# Shared style (matches 1-a)
NOTE_COLOR = "#FCAA6F"      # light orange  (temp=0.3)
PATIENT_COLOR = "#7BB8D4"   # light blue    (temp=0.7)
sns.set_theme(style="ticks", context="paper", font_scale=1.1)

temp07_note_level_confusion_matrix = {
    "true_negatives": 2733, "false_positives": 935,
    "false_negatives": 174, "true_positives": 238,
}
temp03_note_level_confusion_matrix = {
    "true_negatives": 2689, "false_positives": 896,
    "false_negatives": 174, "true_positives": 232,
}
temp07_patient_level_confusion_matrix = {
    "true_negatives": 43, "false_positives": 7,
    "false_negatives": 10, "true_positives": 40,
}
temp03_patient_level_confusion_matrix = {
    "true_negatives": 43, "false_positives": 7,
    "false_negatives": 10, "true_positives": 40,
}

def compute_metrics(confusion_matrix):
    tn = confusion_matrix["true_negatives"]
    fp = confusion_matrix["false_positives"]
    fn = confusion_matrix["false_negatives"]
    tp = confusion_matrix["true_positives"]
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1_score  = 2 * precision * recall / (precision + recall)
    f3_score  = 10 * precision * recall / (9 * precision + recall)
    return accuracy, precision, recall, f1_score, f3_score

temp03_note_level_metrics    = compute_metrics(temp03_note_level_confusion_matrix)
temp07_note_level_metrics    = compute_metrics(temp07_note_level_confusion_matrix)
temp03_patient_level_metrics = compute_metrics(temp03_patient_level_confusion_matrix)
temp07_patient_level_metrics = compute_metrics(temp07_patient_level_confusion_matrix)

bar_width = 0.45
fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.5), gridspec_kw={'width_ratios': [6, 5]})

# --- Note-level ---
note_labels = ["Accuracy", "Precision", "Sensitivity", "F1 Score", "F3 Score", "Processing Rate"]
x = np.arange(len(note_labels))
bars1 = axes[0].bar(x - bar_width/2, list(temp03_note_level_metrics) + [3991/4489], bar_width,
                    label="temp=0.3", color=NOTE_COLOR, alpha=0.88, edgecolor='#444444', linewidth=0.8)
bars2 = axes[0].bar(x + bar_width/2, list(temp07_note_level_metrics) + [4080/4489], bar_width,
                    label="temp=0.7", color=PATIENT_COLOR, alpha=0.88, edgecolor='#444444', linewidth=0.8)
for bar in bars1 + bars2:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, height + 0.006, f'{height:.3f}',
                 ha='center', va='bottom', fontsize=7, color='#333333')
axes[0].set_title("Note-Level Metrics Comparison", fontsize=13, pad=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(note_labels, rotation=25, ha='center', fontsize=10)
axes[0].tick_params(axis='y', labelsize=11)
axes[0].set_ylabel("Score", fontsize=12)
axes[0].set_ylim(0, 1.13)
axes[0].set_yticks(np.linspace(0, 1.0, 6))
axes[0].yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.65)
axes[0].set_axisbelow(True)
axes[0].legend(frameon=True, framealpha=0.85, edgecolor='lightgray', fontsize=9)
sns.despine(ax=axes[0])

# --- Patient-level ---
patient_labels = ["Accuracy", "Precision", "Sensitivity", "F1 Score", "F3 Score"]
x = np.arange(len(patient_labels))
bars3 = axes[1].bar(x - bar_width/2, temp03_patient_level_metrics, bar_width,
                    label="temp=0.3", color=NOTE_COLOR, alpha=0.88, edgecolor='#444444', linewidth=0.8)
bars4 = axes[1].bar(x + bar_width/2, temp07_patient_level_metrics, bar_width,
                    label="temp=0.7", color=PATIENT_COLOR, alpha=0.88, edgecolor='#444444', linewidth=0.8)
for bar in bars3 + bars4:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, height + 0.001, f'{height:.3f}',
                 ha='center', va='bottom', fontsize=7, color='#333333')
axes[1].set_title("Patient-Level Metrics Comparison", fontsize=13, pad=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(patient_labels, rotation=25, ha='center', fontsize=10)
axes[1].tick_params(axis='y', labelsize=11)
axes[1].set_ylabel("Score", fontsize=12)
axes[1].set_ylim(0.7, 0.926)
axes[1].set_yticks(np.linspace(0.7, 0.9, 6))
axes[1].yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.65)
axes[1].set_axisbelow(True)
axes[1].legend(frameon=True, framealpha=0.85, edgecolor='lightgray', fontsize=9)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig("plots/4-d-temperature.pdf", bbox_inches='tight')
