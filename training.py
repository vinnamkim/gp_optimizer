import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from torch import optim
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from dqn import DQN
from mlnn import MLNN
from replay_memory import ReplayMemory, Transition

def optimize_dqn(policy_net, target_net, replay_memory, optimizer, batch_size, gamma):
    if len(replay_memory) < batch_size:
        return
    
    transitions = replay_memory.sample(batch_size)
    
    batch = Transition(*zip(*transitions))

    q_values = policy_net(batch.state).gather(1, batch.action)
    expected_q_values = (target_net(batch.next_state).max(1)[0].detach() * gamma) + batch.reward

    loss = F.smooth_l1_loss(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    
    optimizer.step()
    
    return
