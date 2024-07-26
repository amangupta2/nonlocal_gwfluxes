import numpy as np

# Read the text file
with open('./test.txt', 'r') as file:
    lines = file.readlines()

# Initialize lists to store the losses
training_losses = []
testing_losses = []

# Extract losses from each line
for line in lines:
    parts = line.split(',')
    training_loss = float(parts[2].split(':')[1].strip())
    testing_loss = float(parts[3].split(':')[1].strip())
    training_losses.append(training_loss)
    testing_losses.append(testing_loss)

# Convert lists to numpy arrays
training_losses = np.array(training_losses)
testing_losses = np.array(testing_losses)

print("Training Losses:", training_losses)
print("Testing Losses:", testing_losses)
