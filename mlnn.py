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

    def init_variables(self):
        for m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def get_replays(self):
        return self.replays
    
    def get_eps_threshold(self):
        return EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.steps_done / EPS_DECAY)

    def forward(self, x, dqn):
        if self.training:
            eps_threshold = self.get_eps_threshold()
        else:
            eps_threshold = -1.0

        self.replays = []
        
        outputs = self.start_layer(x)

        for layers, batch_norm in zip(self.layers, self.batch_norms):
            #print(outputs.shape)
            actions = _get_action(dqn, outputs)
            next_outputs = []
            
            for j in range(len(actions)):
                state = outputs[j]
                actions[j] = actions[j] if random.random() > eps_threshold else torch.tensor(random.randrange(len(layers)))
                next_state = layers[actions[j]](state.reshape([1, -1]))
                next_outputs.append(next_state)
            
            next_outputs = torch.stack(next_outputs).squeeze(1)
            next_outputs = batch_norm(next_outputs)
            
            for i in range(len(outputs)):
                state = outputs[i]
                action = actions[i]
                next_state = next_outputs[i]
                self.replays.append(Transition(state, action, next_state, 0.))
                
            #print(outputs.shape)
        
        outputs = self.end_layer(outputs)

        self.steps_done += 1

        return outputs

    def update_replays(self, loss):
        reward = (1 - loss.detach()).clamp(min=0.)

        for i, replay in enumerate(self.replays):
            self.replays[i] = Transition(
                replay.state.detach(), 
                replay.action, 
                replay.next_state.detach(), 
                reward
            )
