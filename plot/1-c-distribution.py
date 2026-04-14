import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Shared style (matches 1-a and 1-b)
NOTE_COLOR = "#FCAA6F"    # light orange
PATIENT_COLOR = "#7BB8D4"  # light blue
sns.set_theme(style="ticks", context="paper", font_scale=1.1)

# Load the data
final_df = pd.read_csv("data/notes.csv")

fig, ax = plt.subplots(figsize=(5., 4.5))

palette = {
    "TP": NOTE_COLOR, "TN": NOTE_COLOR,
    "FP": PATIENT_COLOR, "FN": PATIENT_COLOR,
}

sns.boxplot(x='category', y='final_probability', data=final_df,
            palette=palette, width=0.5, linewidth=0.9,
            flierprops=dict(marker='o', markersize=3, alpha=0.4), ax=ax)
sns.stripplot(x='category', y='final_probability', data=final_df,
              color='#333333', alpha=0.2, size=2, jitter=0.25, ax=ax)

ax.set_title('Distribution of Probabilities by Classification Category', fontsize=12, pad=10)
ax.set_xlabel('Category')
ax.set_ylabel('Probability')
ax.yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.65)
ax.set_axisbelow(True)
sns.despine()

plt.tight_layout()
plt.savefig("plots/1-c-distribution.pdf", bbox_inches='tight')

# Print summary statistics for each category
print("\nProbability Statistics by Category:")
print(final_df.groupby('category')['final_probability'].describe())
