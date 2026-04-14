import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Shared style (matches 1-b)
sns.set_theme(style="ticks", context="paper", font_scale=1.1)

llama_70b_note_level = {
    "True Negatives": 3847, "False Positives": 123,
    "False Negatives": 190, "True Positives": 228,
}
llama_70b_patient_level = {
    "True Negatives": 48, "False Positives": 2,
    "False Negatives": 15, "True Positives": 35,
}
llama_8b_note_level = {
    "True Negatives": 2774, "False Positives": 667,
    "False Negatives": 162, "True Positives": 215,
}
llama_8b_patient_level = {
    "True Negatives": 43, "False Positives": 7,
    "False Negatives": 13, "True Positives": 37,
}
mistral_note_level = {
    "True Negatives": 1639, "False Positives": 2372,
    "False Negatives": 92,  "True Positives": 330,
}
mistral_patient_level = {
    "True Negatives": 10,  "False Positives": 40,
    "False Negatives": 3,  "True Positives": 47,
}
llama3_note_level = {
    "True Negatives": 2499, "False Positives": 1264,
    "False Negatives": 140, "True Positives": 262,
}
llama3_patient_level = {
    "True Negatives": 35, "False Positives": 15,
    "False Negatives": 5, "True Positives": 45,
}
llama2_note_level = {
    "True Negatives": 192, "False Positives": 943,
    "False Negatives": 10, "True Positives": 77,
}
llama2_patient_level = {
    "True Negatives": 1,  "False Positives": 49,
    "False Negatives": 2, "True Positives": 48,
}

matrices = [
    llama2_patient_level, mistral_patient_level, llama3_patient_level,
    llama_8b_patient_level, llama_70b_patient_level,
]
model_names = [
    "Llama 2-7B-Chat", "Mistral 7B-Instruct-v0.3", "Llama 3-8B-Instruct",
    "Llama 3.1-8B-Instruct", "Llama 3.1-70B-Instruct",
]

fig, axs = plt.subplots(1, 5, figsize=(16, 3.5))

for ax, cm, name in zip(axs, matrices, model_names):
    matrix = np.array([
        [cm["True Negatives"], cm["False Positives"]],
        [cm["False Negatives"], cm["True Positives"]],
    ])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                vmin=0, vmax=matrix.max() * 2.5,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax, linewidths=0)
    ax.set_title(name, fontsize=18, pad=6)
    ax.set_xlabel("Predicted Label", fontsize=15)
    ax.set_ylabel("True Label", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    for t in ax.texts:
        t.set_fontsize(18)
    # Uniform grid lines matching 1-b
    ax.axhline(1, color='#444444', linewidth=2)
    ax.axvline(1, color='#444444', linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('#444444')

plt.tight_layout()
plt.savefig('plots/4-c-model-confusion-matrices.pdf', bbox_inches='tight')
