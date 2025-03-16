import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Environment Setup
n_channels = 5 # Number of channels
n_episodes = 10000

# Scenario 1
# true_probs = [0.1, 0.4, 0.6, 0.3, 0.9]
# Scenario 2
true_probs = [0.5, 0.9, 0.8, 0.6]
# Scenario 3
# true_probs = [0.9, 0.2, 0.5, 0.3, 0.8]

# Q-Learning Parameters
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

# Initialize Q-table for Q-learning
Q_table = np.zeros(n_channels)

# DQN Hyperparameters
alpha_dqn = 0.0005
batch_size = 128
memory_size = 50000
target_update_freq = 20

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Define the neural network layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),  # Input layer
            nn.ReLU(),                 # Activation function
            nn.Linear(16, output_dim)  # Output layer
)
    def forward(self, x):
        # Define forward propagation
        return self.model(x)

# Initialize Neural Networks and optimizer for DQN
dqn = DQN(n_channels, n_channels)
target_dqn = DQN(n_channels, n_channels)


target_dqn.load_state_dict(dqn.state_dict())


# Complete network initialization and optimizer setup
optimizer = optim.Adam(dqn.parameters(), lr=alpha_dqn)
criterion = nn.MSELoss()

# Experience Replay Memory for DQN
replay_memory = deque(maxlen=memory_size)

# Lists to store rewards
rewards_qlearning = []
rewards_dqn = []

# Q-Learning Training Loop
for episode in range(n_episodes):
    # Action selection (epsilon-greedy) for Q-Learning
    random_float = random.uniform(0, 1) # a number in range [0,1)
    chosen_channel = 0 # Default channel is 0
    if random_float <= epsilon:
        # Act randomnly
        chosen_channel = random.randint(0, n_channels-1) # a number in range [0, n-1]
    else:
        # Act on current policy
        current_biggest = Q_table[0] # Default channel is 0
        for i in range(1, n_channels):
            if (Q_table[i] > current_biggest):
                current_biggest = Q_table[i]
                chosen_channel = i

    # Calculate the reward on the chosen channel
    random_float2 = random.uniform(0, 1) # a number in range [0,1)
    reward = 0
    if (random_float2 <= true_probs[chosen_channel]):
        reward = 1

    # Decay epsilon
    epsilon = epsilon * epsilon_decay

    # Ensure epsilon doesn't go below the min (want to retain a small amount of randomness)
    if epsilon < epsilon_min:
        epsilon = epsilon_min

    # Update Q-table
    Q_table[chosen_channel] = Q_table[chosen_channel] + alpha * (reward - Q_table[chosen_channel])

    # Append reward to rewards_qlearning
    rewards_qlearning.append(reward)

# Reset epsilon for DQN
epsilon = 1.0


# DQN Training Loop
for episode in range(n_episodes):
    # Generate a random initial state (one-hot encoded)
    state = np.eye(n_channels)[np.random.choice(n_channels)]
    state_tensor = torch.FloatTensor(state)

    # Action selection (epsilon-greedy)
    if np.random.rand() < epsilon:
        action = np.random.choice(n_channels)  # Explore
    else:
        with torch.no_grad():
            action = torch.argmax(dqn(state_tensor)).item()  # Exploit

    # Calculate reward
    reward = 1 if np.random.rand() < true_probs[action] else 0

    # Update the target Q-values
    target = dqn(state_tensor).clone().detach()
    target[action] = reward

    # Compute the loss and update the model
    output = dqn(state_tensor)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network periodically
    if episode % target_update_freq == 0:
        target_dqn.load_state_dict(dqn.state_dict())

    # Decay epsilon
    epsilon = epsilon * epsilon_decay

    # Ensure epsilon doesn't go below the min (want to retain a small amount of randomness)
    if epsilon < epsilon_min:
        epsilon = epsilon_min

    # Append reward to rewards_dqn
    rewards_dqn.append(reward)


print("After ", n_episodes, " the Q table has generated the probabilities: ", Q_table) # Debug

# Select a specific state (e.g., state 2)
state_index = 0 #choosen state
state = np.eye(n_channels)[state_index]  # One-hot encode the state
state_tensor = torch.FloatTensor(state)  # Convert to a PyTorch tensor

# Get Q-values for the selected state
with torch.no_grad():  # Ensure no gradients are computed
    q_values_for_state = dqn(state_tensor)

# Print the Q-values
print("After ", n_episodes, " the Deep Qtable has generated the probabilities: ",q_values_for_state.numpy())

""" To get values of all the states together use the code below
with torch.no_grad():
    print("\nLearned Q-values:")
    q_values = dqn(torch.eye(n_channels))  # Pass all states through the model
    print(q_values.numpy())  # Print Q-values

print("After ", n_episodes, " the Deep Qtable has generated the probabilities: ", Q_table) # Debug

"""

# Moving Average Calculation
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Compute moving averages
window_size = 1000
q_learning_ma = moving_average(rewards_qlearning, window_size)
dqn_ma = moving_average(rewards_dqn, window_size)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(q_learning_ma, label="Q-learning", linewidth=2)
plt.plot(dqn_ma, label="Deep Q-Network", linestyle="dashed", linewidth=2)
plt.xlabel("Episodes")
plt.ylabel("Moving Average Reward")
plt.title("Performance of Q-learning vs Deep Q-Network")
plt.legend()
plt.grid(True)
plt.show()
