import torch
import torch.nn.functional as F
from replay_memory import Transition

TAU = 0.001

def _clamp_params(net):
    for param in net.parameters():
        param.grad.data.clamp_(-1, 1)
        
def optimize_dqn(policy_net, target_net, replay_memory, optimizer, batch_size, gamma):
    if len(replay_memory) < batch_size:
        return
    
    transitions = replay_memory.sample(batch_size)
    
    batch = Transition(*zip(*transitions))
    
    state = torch.stack(batch.state)
    action = torch.stack(batch.action).reshape([-1, 1])
    next_state = torch.stack(batch.next_state)
    reward = torch.stack(batch.reward)
    
    q_values = policy_net(state).gather(1, action.reshape([-1, 1]).cuda()).squeeze()
    #print(batch.reward)
    expected_q_values = (target_net(next_state).max(1)[0].detach() * gamma) + reward

    loss = F.smooth_l1_loss(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    _clamp_params(policy_net)
    optimizer.step()
    
    for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU*local_param.data + (1.0 - TAU) * target_param.data)
    
    return loss
