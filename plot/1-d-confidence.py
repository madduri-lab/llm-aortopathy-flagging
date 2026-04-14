import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Shared style (matches 1-a to 1-c)
sns.set_theme(style="ticks", context="paper", font_scale=1.1)

base_confusion_matrix =          {"true_positive": 42, "false_positive": 5,  "true_negative": 45, "false_negative": 8}
threshold_05_confusion_matrix =  {"true_positive": 43, "false_positive": 5,  "true_negative": 45, "false_negative": 7}
threshold_06_confusion_matrix =  {"true_positive": 42, "false_positive": 5,  "true_negative": 45, "false_negative": 8}
threshold_07_confusion_matrix =  {"true_positive": 42, "false_positive": 6,  "true_negative": 44, "false_negative": 8}
threshold_08_confusion_matrix =  {"true_positive": 42, "false_positive": 6,  "true_negative": 44, "false_negative": 8}
threshold_09_confusion_matrix =  {"true_positive": 42, "false_positive": 7,  "true_negative": 43, "false_negative": 8}
threshold_095_confusion_matrix = {"true_positive": 41, "false_positive": 7,  "true_negative": 43, "false_negative": 9}
rag_only_confusion_matrix =      {"true_positive": 39, "false_positive": 7,  "true_negative": 43, "false_negative": 11}

def compute_metrics(cm):
    tp, fp, tn, fn = cm["true_positive"], cm["false_positive"], cm["true_negative"], cm["false_negative"]
    accuracy     = (tp + tn) / (tp + tn + fp + fn)
    precision    = tp / (tp + fp)
    recall       = tp / (tp + fn)
    specificity  = tn / (tn + fp)
    f1           = 2 * precision * recall / (precision + recall)
    f3           = 10 * precision * recall / (9 * precision + recall)
    return accuracy, precision, recall, specificity, f1, f3

data = {
    "0.0\n(Base Model)":       compute_metrics(base_confusion_matrix),
    "0.5":  compute_metrics(threshold_05_confusion_matrix),
    "0.6":  compute_metrics(threshold_06_confusion_matrix),
    "0.7":  compute_metrics(threshold_07_confusion_matrix),
    "0.8":  compute_metrics(threshold_08_confusion_matrix),
    "0.9":  compute_metrics(threshold_09_confusion_matrix),
    "0.95": compute_metrics(threshold_095_confusion_matrix),
    "1.0\n(All RAG)":          compute_metrics(rag_only_confusion_matrix),
}

metric_labels = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score", "F3 Score"]
# warm orange + light blue + muted extras, consistent with 1-a/1-c palette
colors = ["#FCAA6F", "#7BB8D4", "#A8D8A8", "#C9A8D4", "#F4C6C6", "#D4B8A8"]

fig, ax = plt.subplots(figsize=(4.5, 4))

x_labels = list(data.keys())
for i, (metric, color) in enumerate(zip(metric_labels, colors)):
    values = [x[i] for x in data.values()]
    ax.plot(x_labels, values, marker='o', markersize=5, linewidth=1.6,
            label=metric, color=color)

ax.set_title("Patient Level Metrics by Confidence Threshold", fontsize=11, pad=40)
ax.set_ylabel("Score")
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=0, ha='center')
ax.set_xlabel("Confidence Threshold")

ax.yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.65)
ax.set_axisbelow(True)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3,
          frameon=False, fontsize=9)
sns.despine()

plt.tight_layout()
plt.savefig("plots/1-d-confidence.pdf", bbox_inches='tight')
