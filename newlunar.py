import gymnasium as gym
import torch
from pathlib import Path

# Load the trained model
class DQN(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)

# Function to evaluate and save video
def save_successful_landing_video():
    env = gym.make("LunarLander-v3", continuous=False, render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, "successful_landings", episode_trigger=lambda x: True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    dqn = DQN(state_dim, action_dim)
    dqn.load_state_dict(torch.load("dqn_lunar_lander.pth"))
    dqn.eval()

    for episode in range(10):  # Run 10 episodes for video capture 
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False

        while not done:
            with torch.no_grad():
                action = torch.argmax(dqn(state)).item()
            next_state, reward, done, _, _ = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    env.close()

if __name__ == "__main__":
    save_successful_landing_video()
    # 