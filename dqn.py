# run this using - python /Users/killuaa/Desktop/RL/DQN/dqn.py

import gymnasium as gym
from train import DQN
import torch
from replay_buffer import *
import itertools
import yaml
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, hyperparameters):
        with open('/Users/killuaa/Desktop/RL/DQN/parameter.yml', 'r') as file:

            all_hyperparameters = yaml.safe_load(file)
            parameter = all_hyperparameters[hyperparameters]

        self.replay_buffer_size = parameter['replay_buffer_size']
        self.batch_size = parameter['batch_size']
        self.epsilon_initial = parameter['epsilon_initial']
        self.epsilon_decay = parameter['epsilon_decay']
        self.epsilon_min = parameter['epsilon_min']
        self.learning_rate = parameter['learning_rate']
        self.gamma = parameter['gamma']
        self.network_sync_rate = parameter['network_sync_rate']

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None
            
    
    def run(self, is_training=True, render = False):

        env = gym.make("LunarLander-v3", render_mode="human" if render else None)

        num_actions = env.observation_space.shape[0]
        num_states = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        reward_per_episode = []
        epsilon_history = []

        if is_training:
            memory = ReplayBuffer(self.replay_buffer_size)

            epsilon = self.epsilon_initial

            target_dqn = policy_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            # policy network optimizer Adam, can changed accordingly
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            terminated = False
            episode_reward = 0.0

            while not terminated :
                if is_training and np.random.rand() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64).to(device)
                else:
                    with torch.no_grad():
                        # convert state to tensor
                        action = policy_dqn(state).argmax()
                        action = torch.clamp(action,0,3)
                          
                new_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                # convert new_state and reward to tensor
                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)

                if is_training:
                    memory.append((state ,new_state, action, reward, terminated))

                    step_count += 1

                state = new_state

            reward_per_episode.append(episode_reward)

            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            if len(memory)>self.batch_size:
                mini_batch = memory.sample(self.batch_size)
                self.optimize(policy_dqn, target_dqn, mini_batch)

                if step_count>self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

    def optimize(self, policy_dqn, target_dqn, mini_batch):
        for state, new_state, action, reward, terminated in mini_batch:
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target = reward + self.gamma * target_dqn(new_state).max()

            predicted_q = policy_dqn(state)

            # calculate loss for whole mini batch
            loss = self.loss_fn(predicted_q, target)

            # optimize the model
            self.optimizer.zero_grad() # clear gradients
            loss.backward() # computate gradients
            self.optimizer.step() # update weights

        

        
if __name__ == "__main__":
    agent = Agent("LunarLander")
    agent.run(is_training=True,render=True)


