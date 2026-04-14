note_level_confusion_matrix = {
    "true_negatives": 14234,
    "false_positives": 4302,
    "false_negatives": 808,
    "true_positives": 1376
}

patient_level_confusion_matrix = {
    "true_negatives": 209,
    "false_positives": 41,
    "false_negatives": 42,
    "true_positives": 207
}

def compute_metrics(confusion_matrix):
    true_negatives = confusion_matrix["true_negatives"]
    false_positives = confusion_matrix["false_positives"]
    false_negatives = confusion_matrix["false_negatives"]
    true_positives = confusion_matrix["true_positives"]

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    f1_score = 2 * precision * recall / (precision + recall)
    f3_score = 10 * precision * recall / (9 * precision + recall)

    return accuracy, precision, recall, specificity, f1_score, f3_score

note_level_metrics = compute_metrics(note_level_confusion_matrix)
patient_level_metrics = compute_metrics(patient_level_confusion_matrix)

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# Shared style
NOTE_COLOR = "#FCAA6F"    # light orange
PATIENT_COLOR = "#7BB8D4"  # light blue
sns.set_theme(style="ticks", context="paper", font_scale=1.1)

metrics_labels = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score", "F3 Score"]
bar_width = 0.38
x = np.arange(len(metrics_labels))

fig, ax = plt.subplots(figsize=(6, 4.5))

bars1 = ax.bar(x - bar_width/2, note_level_metrics, bar_width, label="Note Level",
               color=NOTE_COLOR, alpha=0.88, edgecolor='#444444', linewidth=0.8)
bars2 = ax.bar(x + bar_width/2, patient_level_metrics, bar_width, label="Patient Level",
               color=PATIENT_COLOR, alpha=0.88, edgecolor='#444444', linewidth=0.8)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.006, f'{height:.3f}',
            ha='center', va='bottom', fontsize=7, color='#333333')

ax.set_title("Note and Patient Level Prediction Metrics", fontsize=12, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, rotation=0, ha='center')
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.65)
ax.set_axisbelow(True)
ax.legend(frameon=True, framealpha=0.85, edgecolor='lightgray', fontsize=9)
sns.despine()

plt.tight_layout()
plt.savefig("plots/1-a-metrics-final.pdf", bbox_inches='tight')
