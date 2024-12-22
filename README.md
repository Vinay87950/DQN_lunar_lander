# 🚀 Reinforcement Learning with DQN for Lunar Lander

This repository contains a Deep Q-Network (DQN) implementation for the Lunar Lander environment from OpenAI Gymnasium. The agent is trained to land a spaceship safely on the lunar surface by learning through reinforcement learning.

---

## 📂 Project Structure
📁 DQN
├── load_and_play.py # Script to load a trained model and play episodes
├── trained_lunar_lander.pth # Saved model checkpoint
├── parameter.yml # Configuration for the DQN agent
├── dqn.py # DQN implementation


Set up a virtual environment (recommended):

python3 -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate       


▶️ How to Run
1. Train the Agent
If training is included in your project, provide instructions for training the agent. For example:


python train.py
2. Load and Play
To load a trained model and play episodes:


python load_and_play.py
You can adjust the number of episodes by modifying the num_episodes parameter in the script.
