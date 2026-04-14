# Recreate the plots with updated titles and legends to the side

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
============================================================
PATIENT-LEVEL METRICS BY THRESHOLD
============================================================
           TP  FP  TN  FN  Accuracy  Sensitivity  Specificity  Precision     F1     F3
Threshold
0.1000     45  42   8   5    0.5300       0.9000       0.1600     0.5172 0.6569 0.8380
0.2000     45  26  24   5    0.6900       0.9000       0.4800     0.6338 0.7438 0.8637
0.3000     42  13  37   8    0.7900       0.8400       0.7400     0.7636 0.8000 0.8317
0.4000     40   6  44  10    0.8400       0.8000       0.8800     0.8696 0.8333 0.8065
0.5000     39   3  47  11    0.8600       0.7800       0.9400     0.9286 0.8478 0.7927
0.6000     35   1  49  15    0.8400       0.7000       0.9800     0.9722 0.8140 0.7202
0.7000     28   1  49  22    0.7700       0.5600       0.9800     0.9655 0.7089 0.5846
0.8000     23   1  49  27    0.7200       0.4600       0.9800     0.9583 0.6216 0.4852
0.9000     18   0  50  32    0.6800       0.3600       1.0000     1.0000 0.5294 0.3846
"""

# Shared style (matches 1-a to 1-d)
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
NOTE_COLOR = "#FCAA6F"    # light orange
PATIENT_COLOR = "#7BB8D4"  # light blue
COLORS_METRICS = ["#FCAA6F", "#7BB8D4", "#A8D8A8", "#C9A8D4", "#F4C6C6", "#D4B8A8"]
COLORS_CM      = [NOTE_COLOR, PATIENT_COLOR, "#A8D8A8", "#C9A8D4"]

df = pd.DataFrame({
    "Threshold":   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "TP":          [45, 45, 42, 40, 39, 35, 28, 23, 18],
    "FP":          [42, 26, 13,  6,  3,  1,  1,  1,  0],
    "TN":          [ 8, 24, 37, 44, 47, 49, 49, 49, 50],
    "FN":          [ 5,  5,  8, 10, 11, 15, 22, 27, 32],
    "Accuracy":    [0.53, 0.69, 0.79, 0.84, 0.86, 0.84, 0.77, 0.72, 0.68],
    "Sensitivity": [0.90, 0.90, 0.84, 0.80, 0.78, 0.70, 0.56, 0.46, 0.36],
    "Specificity": [0.16, 0.48, 0.74, 0.88, 0.94, 0.98, 0.98, 0.98, 1.00],
    "Precision":   [0.5172, 0.6338, 0.7636, 0.8696, 0.9286, 0.9722, 0.9655, 0.9583, 1.0000],
    "F1 Score":    [0.6569, 0.7438, 0.8000, 0.8333, 0.8478, 0.8140, 0.7089, 0.6216, 0.5294],
    "F3 Score":    [0.8380, 0.8637, 0.8317, 0.8065, 0.7927, 0.7202, 0.5846, 0.4852, 0.3846],
})

# Plot 1: Patient-Level Metrics by Consensus Threshold
fig, ax = plt.subplots(figsize=(4.8, 4))
for col, color in zip(["Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score", "F3 Score"], COLORS_METRICS):
    ax.plot(df["Threshold"], df[col], marker='o', markersize=5,
            linewidth=1.6, label=col, color=color)

ax.set_title("Patient Level Metrics by Consensus Threshold", fontsize=11, pad=40)
ax.set_xlabel("Consensus Threshold")
ax.set_ylabel("Score")
ax.set_xticks(df["Threshold"])
ax.yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.65)
ax.set_axisbelow(True)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3,
          frameon=False, fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig("plots/1-e-consensus.pdf", bbox_inches='tight')

# Plot 2: Confusion Matrix counts by Consensus Threshold
fig, ax = plt.subplots(figsize=(4.3, 4))
for col, color in zip(["TP", "FP", "TN", "FN"], COLORS_CM):
    ax.plot(df["Threshold"], df[col], marker='o', markersize=5,
            linewidth=1.6, label=col, color=color)

ax.set_title("Confusion Matrix Counts by Consensus Threshold", fontsize=11, pad=30)
ax.set_xlabel("Consensus Threshold")
ax.set_ylabel("Count")
ax.set_xticks(df["Threshold"])
ax.yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.65)
ax.set_axisbelow(True)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4,
          frameon=False, fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig("plots/1-e-consensus-confusion.pdf", bbox_inches='tight')
