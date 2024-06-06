import torch
import pickle
import pathlib
import argparse
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--step_size", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--threshold", type=float, default=0.25)
parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine"])
parser.add_argument("--save_model", type=str, choices=["True", "False"], default="False")
parser.add_argument("--model_path", type=str, default="./embedding_cls_model/nn_model.pth")
parser.add_argument("--do_standardize", type=str, choices=["True", "False"], default="True")
parser.add_argument("--compare_log_reg", type=str, choices=["True", "False"], default="False")
parser.add_argument("--embedding_dir", type=str, default="./embedding/llama-3-8b-lora-genrev-aora-raw-large-mean")
args = parser.parse_args()

# Load embedding and classes
X_train = np.load(f'{args.embedding_dir}/train_embedding.npz')['train_embeddings']
X_val = np.load(f'{args.embedding_dir}/val_embedding.npz')['val_embeddings']

with open(f'{args.embedding_dir}/train_classes.pkl', 'rb') as file:
    y_train = pickle.load(file)
with open(f'{args.embedding_dir}/val_classes.pkl', 'rb') as file:
    y_val = pickle.load(file)

# Standardize the data
if args.do_standardize == "True":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

# Assuming `X_train` and `y_train` are your input and output training data respectively
# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Create a dataset and dataloader for batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Neural network model
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))  # Using sigmoid for the binary classification
        return x

input_size = X_train.shape[1]

# Initialize the model
model = BinaryClassifier(input_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.scheduler == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
elif args.scheduler == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training loop with validation
num_epochs = args.epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()  # Decay the learning rate
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to calculate gradients
        val_losses = []
        predictions = []
        targets = []
        probas = []
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs.squeeze(), val_labels)
            val_losses.append(val_loss.item())
            
            # Store predictions and targets for confusion matrix
            predictions.extend((val_outputs.squeeze() >= args.threshold).float().numpy())
            targets.extend(val_labels.numpy())
            probas.extend(val_outputs.squeeze().numpy())

        avg_val_loss = sum(val_losses) / len(val_losses)
        # Compute confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)
        TN, FP, FN, TP = conf_matrix.ravel()
        # Compute accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy_neg = TN / (TN + FP)
        accuracy_pos = TP / (TP + FN)
        auc_score = roc_auc_score(targets, probas)  # Compute AUC

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, Accuracy Negative: {accuracy_neg:.4f}, Accuracy Positive: {accuracy_pos:.4f}, AUC: {auc_score:.4f}')
    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')

# Compare with logistic regression
if args.compare_log_reg == "True":
    clf = LogisticRegression(random_state=0, C=1.0, max_iter=1000).fit(X_train, y_train) 
    prediction = clf.predict(X_val)
    probabilities = clf.predict_proba(X_val)[:, 1]  # Probabilities of the positive class
    fpr, tpr, thresholds = roc_curve(y_val, probabilities)
    roc_auc = auc(fpr, tpr)
    print(f"LogReg AUC: {roc_auc:.4f}")
    plt.plot(fpr, tpr, lw=2, label='LogReg (area = %0.4f)' % roc_auc)

    model.eval()
    with torch.no_grad():
        probabilities = []
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            probabilities.extend(val_outputs.squeeze().numpy())
        fpr, tpr, thresholds = roc_curve(y_val, probabilities)
        roc_auc = auc(fpr, tpr)
        print(f"NN AUC: {roc_auc:.4f}")
        plt.plot(fpr, tpr, lw=2, label='NN (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    plt.savefig('auc_compare.pdf')

# Save the model
if args.save_model == "True":
    model_dir = pathlib.Path(args.model_path).parent
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    torch.save(model.state_dict(), args.model_path)