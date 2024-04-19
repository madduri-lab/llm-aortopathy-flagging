import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()  

parser.add_argument("--embedding_dir", type=str, default="./embedding/meditron-7b")
parser.add_argument("--cluster_file", type=str, default="meditron-7b.pdf")

args = parser.parse_args()

# Load embedding and classes
X_train = np.load(f'{args.embedding_dir}/train_embedding.npz')['train_embeddings']
X_val = np.load(f'{args.embedding_dir}/val_embedding.npz')['val_embeddings']
X = np.load(f'{args.embedding_dir}/all_embedding.npz')['all_embeddings']

with open(f'{args.embedding_dir}/train_classes.pkl', 'rb') as file:
    train_classes = pickle.load(file)
with open(f'{args.embedding_dir}/val_classes.pkl', 'rb') as file:
    val_classes = pickle.load(file)
with open(f'{args.embedding_dir}/all_classes.pkl', 'rb') as file:
    all_classes = pickle.load(file)


# TSNE dimention reduction visualization
reduced = TSNE(n_components=2, random_state=0).fit_transform(X)
plt.scatter(reduced[:, 0], reduced[:, 1], c=all_classes, cmap='rainbow')
plt.savefig(f"{args.embedding_dir}/{args.cluster_file}")

scaler = StandardScaler()
train_x = scaler.fit_transform(X_train)
test_x = scaler.transform(X_val)

# For a real problem, C should be properly cross validated and the confusion matrix analyzed
clf = LogisticRegression(random_state=0, C=1.0, max_iter=1000).fit(train_x, train_classes) 
prediction = clf.predict(test_x)
target = val_classes
print(f"Precision: {100*np.mean(prediction == target):.2f}%")
cm = confusion_matrix(prediction, target)
print(f"Confusion matrix is {cm}")