import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Define the neural network for approximating Q-values
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# A buffer to store experiences for training
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.index = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.index] = transition
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# Function to train and evaluate the agent
def run(episodes, is_training=True, render=False):
    # Initialize the LunarLander environment
    env = gym.make("LunarLander-v3", continuous=False, render_mode='human' if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create the DQN model and optimizer
    dqn = DQN(state_dim, action_dim)
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer(10000)

    # Hyperparameters
    batch_size = 64
    gamma = 0.99  # Discount factor for future rewards
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    rewards_per_episode = []
    successful_landings = 0

    for episode in tqdm(range(episodes), desc="Running Episodes"):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        done = False

        while not done:
            # Select an action using epsilon-greedy strategy
            if is_training and np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = dqn(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = torch.argmax(q_values).item()

            # Take the selected action
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward

            # Additional reward for successful landing
            if done and reward > 0:
                reward += 100
                successful_landings += 1

            # Store transition in replay buffer
            if is_training:
                buffer.add((state, action, reward, next_state, done))

            state = next_state

            # Train the model using samples from the replay buffer
            if is_training and len(buffer.buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Compute target Q-values
                with torch.no_grad():
                    target_q = rewards + gamma * (1 - dones) * torch.max(dqn(next_states), dim=1)[0]

                # Update Q-values for current states and actions
                current_q = dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

                # Compute loss and backpropagate
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Save the model and plot the results
    if is_training:
        torch.save(dqn.state_dict(), "dqn_lunar_lander.pth")
        plt.plot(rewards_per_episode)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid()
        plt.savefig("lunar_lander_rewards.png")

    env.close()
    print(f"Successful Landings: {successful_landings}/{episodes} ({successful_landings / episodes * 100:.2f}%)")

if __name__ == "__main__":
    # Train the agent
    run(5000, is_training=True, render=False)

    # Evaluate the agent
    run(50, is_training=False, render=True)
