import gymnasium as gym
import torch
import numpy as np
from dqn import Agent, DQN  


def load_and_play(model_path, config_path, num_episodes=5):
    # Setup environment
    env = gym.make("LunarLander-v3", render_mode="human")

    # Initialize agent
    agent = Agent(config_path)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent.initialize_networks(num_states, num_actions)
    
    # Load the saved model
    checkpoint = torch.load(model_path)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print("Model loaded successfully!")
    
    # Set networks to evaluation mode
    agent.policy_net.eval()
    agent.target_net.eval()
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(agent.policy_net.device)
        episode_reward = 0
        terminated = False
        
        while not terminated:
            # Get action from policy network
            with torch.no_grad():
                action = agent.policy_net(state).argmax().item()
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            # Update state
            state = torch.tensor(next_state, dtype=torch.float32).to(agent.policy_net.device)
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")

    return total_rewards

if __name__ == "__main__":
    model_path = '/Users/killuaa/Desktop/RL/DQN/trained_lunar_lander.pth'
    config_path = '/Users/killuaa/Desktop/RL/DQN/parameter.yml'
    rewards = load_and_play(model_path, config_path, num_episodes=5)