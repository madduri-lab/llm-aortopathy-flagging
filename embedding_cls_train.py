import torch
import pickle
import argparse
import numpy as np
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()  

parser.add_argument("--embedding_dir", type=str, default="./embedding/meditron-7b")
parser.add_argument("--do_standardize", type=str, choices=["True", "False"], default="True")

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


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
print(f'Input size: {input_size}')

# Initialize the model
model = BinaryClassifier(input_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to calculate gradients
        val_losses = []
        correct = 0
        total = 0
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs.squeeze(), val_labels)
            val_losses.append(val_loss.item())
            
            # Calculate accuracy
            predicted = (val_outputs.squeeze() >= 0.5).float()  # Threshold probabilities to get binary predictions
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()

        avg_val_loss = sum(val_losses) / len(val_losses)
        accuracy = correct / total

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}')
