import gymnasium as gym
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import yaml
from replay_buffer import ReplayBuffer
from train import DQN

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.params = yaml.safe_load(file)["LunarLander"]

        self.replay_buffer = ReplayBuffer(self.params['replay_buffer_size'])
        self.batch_size = self.params['batch_size']
        self.epsilon = self.params['epsilon_initial']
        self.epsilon_decay = self.params['epsilon_decay']
        self.epsilon_min = self.params['epsilon_min']
        self.gamma = self.params['gamma']
        self.network_sync_rate = self.params['network_sync_rate']
        self.replay_ratio = self.params['replay_ratio']

        self.loss_fn = torch.nn.MSELoss()
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

    def initialize_networks(self, num_states, num_actions):
        self.policy_net = DQN(num_states, num_actions).to(device)
        self.target_net = DQN(num_states, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.params['learning_rate'])

    def select_action(self, state, is_training):
        if is_training and np.random.rand() < self.epsilon:
            return torch.tensor([[env.action_space.sample()]], dtype=torch.int64, device=device)
        with torch.no_grad():
            q_values = self.policy_net(state)
            return torch.tensor([[q_values.argmax().item()]], dtype=torch.int64, device=device)

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Die sample-Methode gibt bereits die entpackten Tensoren zurück
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Keine erneute Konvertierung zu Tensoren notwendig, da dies bereits im ReplayBuffer geschieht
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # Rest der Methode bleibt gleich
        with torch.no_grad():
            target_q_values = rewards + self.gamma * (1 - dones.float()) * self.target_net(next_states).max(1)[0]

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes, render):
        reward_per_episode = []
        epsilon_history = []
        step_count = 0

        for episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            terminated = False
            episode_reward = 0

            while not terminated:
                if render:
                    env.render()

                # Select action
                action = self.select_action(state, is_training=True)
                
                # Execute action in the environment
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)
                
                # Speichere action als skalaren Wert
                action_scalar = action.item()  # Konvertiere zu einem einzelnen Wert

                # Store transition in replay buffer
                self.replay_buffer.append((state, action_scalar, reward, next_state, terminated))

                state = next_state
                episode_reward += reward.item()
                step_count += 1

                # Optimize the model
                if step_count % self.replay_ratio == 0:
                    self.optimize_model()

                # Sync target network
                if step_count % self.network_sync_rate == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            epsilon_history.append(self.epsilon)
            reward_per_episode.append(episode_reward)

            # Print episode metrics
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.4f}")

        return reward_per_episode, epsilon_history

    def evaluate(self, env, num_episodes, render):
        total_rewards = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            terminated = False
            episode_reward = 0

            while not terminated:
                if render:
                    env.render()

                with torch.no_grad():
                    action = self.policy_net(state).argmax().unsqueeze(0)
                next_state, reward, terminated, truncated, _ = env.step(action.item())

                state = torch.tensor(next_state, dtype=torch.float32).to(device)
                episode_reward += reward

            total_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
        return total_rewards
    
    @staticmethod
    def plot(rewards, epsilons):
        """
    Simple function to plot training rewards and epsilon decay
    """
        plt.figure(figsize=(12, 5))
    
    # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(rewards, label='Rewards')
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
    
    # Plot epsilon decay
        plt.subplot(1, 2, 2)
        plt.plot(epsilons, label='Epsilon', color='green')
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
    
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    config_path = '/Users/killuaa/Desktop/RL/DQN/parameter.yml'
    env = gym.make("LunarLander-v3")

    agent = Agent(config_path)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent.initialize_networks(num_states, num_actions)

    # Train the agent
    rewards, epsilons = agent.train(env, num_episodes=600, render=False)

    Agent.plot(rewards, epsilons)

    # Speichere das trainierte Modell
    model_save_path = '/Users/killuaa/Desktop/RL/DQN/trained_lunar_lander.pth'
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'rewards': rewards,
        'epsilons': epsilons
    }, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate the agent
    agent.evaluate(env, num_episodes=10, render=True)