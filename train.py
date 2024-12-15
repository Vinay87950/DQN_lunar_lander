import torch
from torch import nn
import torch.nn.functional as F


# implenting DQN - fully connected layer
'''
in terms how agent navigates the map - episilon-Greddy algorithm
input layer = state --policy network 
hidden layer - 
output layer = action 
'''
class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == '__main__':
    state_dim = 8
    action_dim = 4

    policy_net = DQN(state_dim, action_dim)
    # input
    state = torch.randn(10, state_dim)
    action = policy_net(state)
    print(action)
