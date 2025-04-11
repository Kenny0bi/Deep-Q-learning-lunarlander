# DQN Lunar Lander – Reinforcement Learning Project

This project implements Deep Q-Learning (DQN) and classical Q-table methods to solve the `LunarLander-v3` environment from OpenAI Gym. The agent is trained using neural networks and a replay buffer, and performance is evaluated via plotted rewards and video recordings of successful landings.

## 🚀 Environment
- `LunarLander-v3` (discrete control)
- Framework: `Gymnasium`  
- DQN implementation with PyTorch

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `DQNlunar.py` | Core Deep Q-Learning implementation with training and evaluation loop |
| `lunar.py` | DQN training with video capture of successful landings |
| `newlunar.py` | Evaluates trained model and saves video using `RecordVideo` |
| `record_lunar.py` | Classic Q-learning variant using discrete Q-table + CSV logging |
| `view_q_table.py` | Tool to visualize and export Q-table as CSV |
| `dqn_lunar_lander.pth` | Trained DQN model weights (PyTorch) |
| `successful_landing.mp4` / `training.mp4` | Sample videos of trained agent performance |

---

## 🧠 Core Features

- DQN agent with:
  - Experience replay
  - Epsilon-greedy action selection
  - Neural network-based Q-function
- Classic tabular Q-learning with state discretization
- Model saving and loading with `.pth`
- Custom reward shaping for successful landings
- Agent behavior recorded and rendered to `.mp4`

---

## ▶️ How to Run

Install dependencies:
```bash
pip install gymnasium[box2d] torch numpy matplotlib tqdm imageio

Train the DQN model: python DQNlunar.py

Evaluate trained agent and record video: python newlunar.py

Classic Q-learning with recording: python record_lunar.py

View or export Q-table: python view_q_table.py
 

Output
dqn_lunar_lander.pth: Trained model

lunar_lander_rewards.png: Reward curve

successful_landing.mp4: Agent landing video

lunar_lander_transitions.csv: Logged state transitions

q_table.csv: Exported Q-table

✍️ Author
Kehinde Obidele
Graduate Student | Health Informatics | Reinforcement Learning


