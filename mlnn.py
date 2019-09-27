from dqn import DQN
from torch import nn
import torch
from replay_memory import Transition
import math
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

def _get_layer(input_dims, output_dims, device):
    return nn.Sequential(
        nn.Linear(input_dims, output_dims),
        nn.ReLU(),
        #nn.BatchNorm1d(output_dims)
    ).to(device)

def _get_action(dqn, x):
    with torch.no_grad():
        return dqn(x).argmax(1)

class MLNN:
    def __init__(self, n_layers, n_width, input_dims, hidden_dims, output_dims, device="cuda"):
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                self.layers.append(
                    _get_layer(input_dims, hidden_dims, device)
                )
            elif i == n_layers - 1:
                self.layers.append(_get_layer(hidden_dims, output_dims, device))
            else:
                self.layers.append(
                    [_get_layer(hidden_dims, hidden_dims, device)
                     for _ in range(n_width)]
                )
        self.steps_done = 0
        self.init_variables()
    
    def get_params(self):
        params = []
        for layers in self.layers:
            if layers is list:
                for m in layers:
                    for param in m.parameters():
                        params.append(param)
            for m in layers:
                for param in m.parameters():
                    params.append(param)
        return params

    def zero_grad(self):
        for layers in self.layers:
            if layers is list:
                for m in layers:
                    m.zero_grad()
            for m in layers:
                m.zero_grad()

    def init_variables(self):
        for layers in self.layers:
            if layers is list:
                for layer in layers:
                    for m in layer:
                        if isinstance(m, nn.Linear):
                            nn.init.kaiming_normal_(m.weight)
                            nn.init.constant_(m.bias, 0)
            else:
                for m in layers:
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, y, dqn, eps_threshold):
        replays = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                outputs = layer(x)
            elif i == len(self.layers) - 1:
                outputs = layer(outputs)
            else:
                actions = _get_action(dqn, outputs)
                next_outputs = []
                for j in range(len(actions)):
                    state = outputs[j]
                    action = actions[j] if random.random() > eps_threshold else torch.tensor(random.randrange(len(layer))).cuda()
                    next_state = layer[action](state.reshape([1, -1]))

                    replays.append(Transition(state, action, next_state.squeeze(), 0.))

                    next_outputs.append(next_state)
                outputs = torch.cat(next_outputs, 0)
        
        return outputs, replays
        
    def train(self, x, y, dqn, criterion):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)

        outputs, replays = self.forward(x, y, dqn, eps_threshold)

        loss = criterion(outputs, y)
        reward = (1 - loss.detach()).clamp(min=0.)

        for i, replay in enumerate(replays):
            replays[i] = Transition(
                replay.state.detach(), 
                replay.action, 
                replay.next_state.detach(), 
                reward
            )

        self.steps_done += 1

        return replays, loss, outputs
