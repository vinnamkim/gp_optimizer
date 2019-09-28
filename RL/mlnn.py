from dqn import DQN
from torch import nn
import torch
from replay_memory import Transition
import math
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

GAMMA = 0.999

def _get_layer(input_dims, output_dims, device):
    return nn.Sequential(
        nn.Linear(input_dims, output_dims),
        nn.ReLU(),
        #nn.BatchNorm1d(output_dims)
    ).to(device)

def _get_action(dqn, x):
    with torch.no_grad():
        return dqn(x).argmax(1)

class MLNN(nn.Module):
    def __init__(self, n_layers, n_width, input_dims, hidden_dims, output_dims, device="cuda"):
        super(MLNN, self).__init__()
        self.start_layer = nn.Sequential(
            _get_layer(input_dims, hidden_dims, device),
            nn.BatchNorm1d(hidden_dims)).to(device)

        self.end_layer = _get_layer(hidden_dims, output_dims, device)

        self.layers = []
        self.batch_norms = []

        for _ in range(n_layers - 2):
            self.layers.append(
                [_get_layer(hidden_dims, hidden_dims, device)
                    for _ in range(n_width)]
            )
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_dims, hidden_dims).to(device)
            )

        self.steps_done = 0
        self.init_variables()
        self.device = device

    def init_variables(self):
        for m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def get_replays(self):
        replays = []
        for i in self.replays:
            replays += self.replays[i]
        return replays
    
    def get_eps_threshold(self):
        return EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.steps_done / EPS_DECAY)

    def forward(self, x, dqn):
        if self.training:
            eps_threshold = self.get_eps_threshold()
        else:
            eps_threshold = -1.0

        self.replays = {}
        
        outputs = self.start_layer(x)
        
        n_layers = len(self.layers)

        for i, layers, batch_norm in zip(range(n_layers), self.layers, self.batch_norms):
            actions = _get_action(dqn, outputs).cpu()
            next_outputs = []
            
            for j in range(len(outputs)):
                state = outputs[j]
                actions[j] = actions[j] if random.random() > eps_threshold else torch.tensor(random.randrange(len(layers)))
                next_state = layers[actions[j]](state.reshape([1, -1]))
                next_outputs.append(next_state)
            
            next_outputs = torch.stack(next_outputs).squeeze(1)
            next_outputs = batch_norm(next_outputs)
            
            self.replays[i] = []
            for j in range(len(outputs)):
                state = outputs[j]
                action = actions[j]
                next_state = next_outputs[j]
                self.replays[i].append(Transition(state, action, next_state, 0.))
            
            #print(outputs.shape)
        
        outputs = self.end_layer(outputs)

        self.steps_done += 1

        return outputs

    def update_replays(self, labels, loss, num_labels):
        reward = (1 - loss.detach()).clamp(min=0.)
        
        y_onehot = torch.IntTensor(len(labels), num_labels)
        y_onehot.zero_()
        y_onehot.scatter_(1, labels.cpu().reshape([-1, 1]), 1)

        A = y_onehot.mm(y_onehot.transpose(0, 1))
        
        for i in self.replays:
            B = torch.IntTensor(len(labels), num_labels)
            B.zero_()
            actions = torch.tensor(
                [replay.action for replay in self.replays[i]], dtype=torch.long).reshape([-1, 1])
            B.scatter_(1, actions, 1)
            B = B.mm(B.transpose(0, 1))
            
            equal_reward = (A * B).float().mean(1)
            diff_reward = ((1 - A) * (1 - B)).float().mean(1)

            for j, replay in enumerate(self.replays[i]):
                self.replays[i][j] = Transition(
                    replay.state.detach(), 
                    replay.action, 
                    replay.next_state.detach(), 
                    reward + (equal_reward[j] + diff_reward[j])
                )

