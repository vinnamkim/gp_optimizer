from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.Linear(hidden_dims, output_dims),
            nn.ReLU(),
            #nn.BatchNorm1d(output_dims)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
