# implenting DQN - fully connected layer
'''
in terms how agent navigates the map - episilon-Greddy algorithm
input layer = state --policy network 
hidden layer - 
output layer = action 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define a more standard DQN architecture with multiple layers and activations
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # Apply layers with ReLU activations and layer normalization
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.fc3(x)  # Output layer with raw Q-values
        return x

    

if __name__ == '__main__':
    state_dim = 8
    action_dim = 4

    policy_net = DQN(state_dim, action_dim)
    # input
    state = torch.randn(10, state_dim)
    action = policy_net(state)
    print(action)
