import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# Shared style (matches 1-a, 4-d)
COLOR_BASE  = "#FCAA6F"   # light orange
COLOR_LARGE = "#7BB8D4"   # light blue
COLOR_SMALL = "#A8D8A8"   # light green
sns.set_theme(style="ticks", context="paper", font_scale=1.1)

base_note_level = {
    "True Negatives": 1436, "False Positives": 344,
    "False Negatives": 103, "True Positives": 122,
}
base_patient_level = {
    "True Negatives": 20, "False Positives": 5,
    "False Negatives": 8, "True Positives": 17,
}
lora_large_note_level = {
    "True Negatives": 392,  "False Positives": 1494,
    "False Negatives": 18,  "True Positives": 209,
}
lora_large_patient_level = {
    "True Negatives": 0,  "False Positives": 25,
    "False Negatives": 0, "True Positives": 25,
}
lora_small_note_level = {
    "True Negatives": 405,  "False Positives": 1240,
    "False Negatives": 20,  "True Positives": 197,
}
lora_small_patient_level = {
    "True Negatives": 0,  "False Positives": 25,
    "False Negatives": 0, "True Positives": 25,
}

def compute_metrics(confusion_matrix):
    tn = confusion_matrix["True Negatives"]
    fp = confusion_matrix["False Positives"]
    fn = confusion_matrix["False Negatives"]
    tp = confusion_matrix["True Positives"]
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1_score  = 2 * precision * recall / (precision + recall)
    f3_score  = 10 * precision * recall / (9 * precision + recall)
    return accuracy, precision, recall, f1_score, f3_score

base_note_level_metrics       = compute_metrics(base_note_level)
base_patient_level_metrics    = compute_metrics(base_patient_level)
lora_large_note_level_metrics = compute_metrics(lora_large_note_level)
lora_large_patient_level_metrics = compute_metrics(lora_large_patient_level)
lora_small_note_level_metrics = compute_metrics(lora_small_note_level)
lora_small_patient_level_metrics = compute_metrics(lora_small_patient_level)

metrics_labels = ["Accuracy", "Precision", "Sensitivity", "F1 Score", "F3 Score"]
bar_width = 0.31
x = np.arange(len(metrics_labels))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

def annotate(ax, bars, offset):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset, f'{height:.3f}',
                ha='center', va='bottom', fontsize=6.5, color='#333333')

# --- Note-level ---
bars1 = axes[0].bar(x - bar_width, base_note_level_metrics, bar_width,
                    label="Base", color=COLOR_BASE, alpha=0.88, edgecolor='#444444', linewidth=0.8)
bars2 = axes[0].bar(x, lora_large_note_level_metrics, bar_width,
                    label="Lora-Large", color=COLOR_LARGE, alpha=0.88, edgecolor='#444444', linewidth=0.8)
bars3 = axes[0].bar(x + bar_width, lora_small_note_level_metrics, bar_width,
                    label="Lora-Small", color=COLOR_SMALL, alpha=0.88, edgecolor='#444444', linewidth=0.8)
annotate(axes[0], bars1 + bars2 + bars3, 0.006)
axes[0].set_title("Note-Level Metrics Comparison", fontsize=12, pad=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_labels, rotation=25, ha='center')
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1.13)
axes[0].yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.65)
axes[0].set_axisbelow(True)
axes[0].legend(frameon=True, framealpha=0.85, edgecolor='lightgray', fontsize=9, loc='upper left')
sns.despine(ax=axes[0])

# --- Patient-level ---
bars4 = axes[1].bar(x - bar_width, base_patient_level_metrics, bar_width,
                    label="Base", color=COLOR_BASE, alpha=0.88, edgecolor='#444444', linewidth=0.8)
bars5 = axes[1].bar(x, lora_large_patient_level_metrics, bar_width,
                    label="Lora-Large", color=COLOR_LARGE, alpha=0.88, edgecolor='#444444', linewidth=0.8)
bars6 = axes[1].bar(x + bar_width, lora_small_patient_level_metrics, bar_width,
                    label="Lora-Small", color=COLOR_SMALL, alpha=0.88, edgecolor='#444444', linewidth=0.8)
annotate(axes[1], bars4 + bars5 + bars6, 0.006)
axes[1].set_title("Patient-Level Metrics Comparison", fontsize=12, pad=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics_labels, rotation=25, ha='center')
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1.13)
axes[1].yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.65)
axes[1].set_axisbelow(True)
axes[1].legend(frameon=True, framealpha=0.85, edgecolor='lightgray', fontsize=9)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig("plots/supp-1-finetune.pdf", bbox_inches='tight')
