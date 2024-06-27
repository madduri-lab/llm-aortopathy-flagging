

import re
import matplotlib.pyplot as plt

# Function to read the log file and extract training loss
def extract_training_loss(log_file_path):
    losses = []
    steps = []
    steps_set = set()

    # Regular expression to match the training loss lines
    loss_pattern = re.compile(r'Training Epoch: \d+/\d+, step (\d+)/\d+ completed \(loss: ([\d\.]+)\)')

    with open(log_file_path, 'r') as file:
        for line in file:
            match = loss_pattern.search(line)
            if match:
                step = int(match.group(1))
                if step not in steps_set:
                    loss = float(match.group(2))
                    steps.append(step)
                    losses.append(loss)
                    steps_set.add(step)

    return steps, losses

# Function to plot the training loss
def plot_training_loss(steps, losses):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linestyle='-', color='b')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('loss.pdf')
    
import numpy as np

def moving_average(losses, window_size=500):
    moving_avg = []
    start = 0
    while start < len(losses):
        end = min(start+window_size, len(losses))
        moving_avg.append(np.mean(losses[start:end]))
        start = end
    return moving_avg

# Read the log file and extract the training loss
log_file_path = 'slurm-3773565-llama3-merge.out'  # Replace with your actual log file path
steps, losses = extract_training_loss(log_file_path)
print(len(steps))
# Plot the training loss
plot_training_loss(steps, losses)
print(moving_average(losses))