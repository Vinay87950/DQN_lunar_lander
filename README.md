# Deep Q-Network (DQN) Implementation  

## Project Overview  
This project implements a **Deep Q-Network (DQN)** to solve reinforcement learning tasks. The model uses a fully connected neural network architecture with features like experience replay and epsilon-greedy exploration. The parameters for training are configurable via a YAML file for flexibility.  

---

## Features  
- Implementation of a **DQN** with a multi-layer perceptron architecture.  
- **Epsilon-Greedy Algorithm** for exploration and exploitation.  
- Modularized components for training, evaluation, and replay buffer handling.  
- YAML-based parameter configuration for easy hyperparameter tuning.  
- Scripts to train, evaluate, and visualize agent performance.

---

## Table of Contents  
1. [Introduction](#introduction)  
2. [Project Structure](#project-structure)   
3. [Usage](#usage)  
4. [How It Works](#how-it-works)  
5. [Results](#results)  
6. [License](#license)  

---

## Introduction  
Reinforcement learning allows agents to learn optimal policies through trial and error. This project demonstrates the **Deep Q-Learning (DQN)** algorithm, which combines Q-Learning with a neural network to approximate the Q-function. The epsilon-greedy algorithm balances exploration and exploitation during training.  

---

## Project Structure  

- **`train.py`**: Implements the training loop for the DQN agent.  
- **`dqn.py`**: Defines the neural network architecture for the DQN.  
- **`replay_buffer.py`**: Manages the experience replay buffer.  
- **`load_and_play.py`**: Script for loading a trained model and evaluating it in an environment.  
- **`parameters.yml`**: YAML file containing configurable training parameters (e.g., learning rate, batch size).  

---

## Usage
- Configuring Parameters
- Training the Model
- Evaluating the Model
- Visualizing Results

---

## How it Works
- **Neural Network Architecture (Defined in dqn.py):**
  
  - **Input Layer:** Encodes the state.
  
  - **Hidden Layers:** Two fully connected layers with ReLU activations and layer normalization for stability.
 
  - **Output Layer:** Outputs raw Q-values for all possible actions.
 
- ** Epsilon-Greedy Algorithm:**

  - Balances exploration (random action selection) and exploitation (choosing the best action)
 
  - Epsilon decays over time for improved convergence
 
- ** Experience Replay:**

  - **Stores transitions **(state, action, reward, next_state)** in a replay buffer




