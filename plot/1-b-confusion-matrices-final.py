import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

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

# Shared style
sns.set_theme(style="ticks", context="paper", font_scale=1.1)

def plot_confusion_matrix_ax(ax, cm, title, cmap_name="Blues"):
    matrix = np.array([[cm["true_negatives"], cm["false_positives"]],
                       [cm["false_negatives"], cm["true_positives"]]])

    sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap_name, cbar=False,
                vmin=0, vmax=matrix.max() * 2.5,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax, linewidths=0)
    ax.set_title(title, fontsize=14, pad=6)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    for t in ax.texts:
        t.set_fontsize(12)
    # Draw only the inner dividing lines; set spines to match
    ax.axhline(1, color='#444444', linewidth=1)
    ax.axvline(1, color='#444444', linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('#444444')

fig, axes = plt.subplots(2, 1, figsize=(2.8, 5.5))

plot_confusion_matrix_ax(axes[0], note_level_confusion_matrix, "Note Level", cmap_name="Oranges")
plot_confusion_matrix_ax(axes[1], patient_level_confusion_matrix, "Patient Level", cmap_name="Blues")

plt.tight_layout(h_pad=2.0)
plt.savefig("plots/1-b-confusion-matrices-final.pdf", bbox_inches='tight')
