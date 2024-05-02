import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

embedding_dirs = [
    "./embedding/meditron-7b",
    "./embedding/meditron-70b",
    "./embedding/llama-2-7b",
    "./embedding/llama-2-7b-lora-small",
    "./embedding/llama-2-7b-lora-large",
    "./embedding/llama-2-13b",
    "./embedding/llama-2-70b",
    "./embedding/llama-3-8b",
    "./embedding/llama-3-8b-lora-small",
    "./embedding/llama-3-70b",
    "./embedding/mistral-7b-v01",
    "./embedding/mistral-7b-v01-lora-small",
    "./embedding/mistral-7b-v01-lora-large",
    "./embedding/mixtral-8x7b",
]

plt.figure(figsize=(10, 8))

for dir in embedding_dirs:
    model_name = dir.split('/')[-1]
    # Load embedding and classes
    X_train = np.load(f'{dir}/train_embedding.npz')['train_embeddings']
    X_val = np.load(f'{dir}/val_embedding.npz')['val_embeddings']
    X = np.load(f'{dir}/all_embedding.npz')['all_embeddings']

    with open(f'{dir}/train_classes.pkl', 'rb') as file:
        train_classes = pickle.load(file)
    with open(f'{dir}/val_classes.pkl', 'rb') as file:
        val_classes = pickle.load(file)
    with open(f'{dir}/all_classes.pkl', 'rb') as file:
        all_classes = pickle.load(file)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(X_train)
    test_x = scaler.transform(X_val)

    # For a real problem, C should be properly cross validated and the confusion matrix analyzed
    clf = LogisticRegression(random_state=0, C=1.0, max_iter=1000).fit(train_x, train_classes) 

    # Predict probabilities for the test data
    probabilities = clf.predict_proba(test_x)[:, 1]  # Probabilities of the positive class

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(val_classes, probabilities)
    roc_auc = auc(fpr, tpr)

    # Compute sensitivity and specificity
    prediction = clf.predict(test_x)
    target = val_classes
    cm = confusion_matrix(prediction, target)
    print(f"\n==============\nFor {model_name}")
    print(f"Confusion matrix is \n{cm}")

    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)  # Also known as recall or true positive rate
    specificity = TN / (TN + FP)  # True negative rate
    # Log the results
    print(f"Sensitivity (Recall or TPR): {sensitivity:.2f}")
    print(f"Specificity (TNR): {specificity:.2f}")
    print(f"Accuracy is {np.mean(prediction == target):.4f}")
    
    plt.plot(fpr, tpr, lw=2, label=model_name+' (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC')
plt.legend(loc="lower right")
plt.savefig('auc.pdf')

