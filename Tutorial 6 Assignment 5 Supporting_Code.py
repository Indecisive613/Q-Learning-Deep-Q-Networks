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
n_channels = 5
n_episodes = 10000
true_probs = [0.1, 0.4, 0.6, 0.3, 0.9]

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
            q_values = dqn(state_tensor)
            action = torch.argmax(q_values).item()  # Exploit

    # Calculate reward
    reward = 1 if np.random.rand() < true_probs[action] else 0

    # Generate the next state (one-hot encoded)
    next_state = np.eye(n_channels)[np.random.choice(n_channels)]
    next_state_tensor = torch.FloatTensor(next_state)

    # Store experience in replay memory
    replay_memory.append((state, action, reward, next_state))

    # Train DQN using replay memory
    if len(replay_memory) >= batch_size:
        # Sample a batch from replay memory
        batch = random.sample(replay_memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Compute Q-values and target Q-values
        current_q_values = dqn(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = target_dqn(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values

        # Compute loss and update DQN
        loss = criterion(current_q_values.squeeze(), target_q_values)
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
    # Currently just adding a number between 0-1 so I can make a graph
    
       
print("After ", n_episodes, " the Qtable has generated the probabilities: ", Q_table) # Debug


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
