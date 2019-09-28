# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from torch import optim
from torch.nn import CrossEntropyLoss


#%%
from dqn import DQN
from torch.nn.functional import smooth_l1_loss
from mlnn import MLNN
from replay_memory import ReplayMemory, Transition
from training import optimize_dqn
import random


#%%
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)


#%%
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


#%%
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


#%%
batch_size = 100
epochs = 200
input_dims = 784
hidden_dims = 333
output_dims = 10
gamma = 0.999
target_update = 10
len_train_dataset = len(train_loader.dataset)


#%%
n_layers = 3
n_width = 2


#%%
policy_net = DQN(hidden_dims, hidden_dims, n_width).cuda()
target_net = DQN(hidden_dims, hidden_dims, n_width).cuda()
model = MLNN(n_layers, n_width, input_dims, hidden_dims, output_dims)


#%%
memory_capacity = 100
replay_memory = ReplayMemory(memory_capacity)


#%%
optimizer = optim.Adam(model.parameters())
dqn_optimizer = optim.Adam(policy_net.parameters())
criterion = CrossEntropyLoss().cuda()


#%%
def get_accuracy(model, dqn):
    model.eval()
    dqn.eval()
    
    correct = 0.
    total = 0.

    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28)).cuda()
        outputs = model(images, dqn)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct+= (predicted.cpu() == labels).sum()

    accuracy = 100 * correct.float() / total
    
    model.train()
    dqn.train()
    
    return accuracy


#%%
avg_mlnn_losses = []
avg_dqn_losses = []

for epoch in range(epochs):
    avg_mlnn_loss = 0.
    avg_dqn_loss = 0.
    
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28)).cuda()
        labels = Variable(labels).cuda()

        outputs = model(images, policy_net)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.update_replays(labels, loss, output_dims)
        
        avg_mlnn_loss += loss.detach()

        for replay in model.get_replays():
            replay_memory.push(replay.state, replay.action, replay.next_state, replay.reward)

        avg_dqn_loss += optimize_dqn(policy_net, target_net, replay_memory, dqn_optimizer, batch_size, gamma)

        if i % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    avg_mlnn_loss /= float(i)
    avg_dqn_loss /= float(i)
    
    avg_mlnn_losses.append(avg_mlnn_loss)
    avg_dqn_losses.append(avg_dqn_loss)
    
    print(epoch, avg_mlnn_loss.data, avg_dqn_loss.data, model.get_eps_threshold())
    
    if epoch % 10 == 0:
        print("accuracy : ", get_accuracy(model, target_net))


#%%
def get_accuracy(model, dqn):
    model.eval()
    dqn.eval()
    
    correct = 0.
    total = 0.
    
    path = []

    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28)).cuda()
        outputs = model(images, dqn)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct+= (predicted.cpu() == labels).sum()
        path.append(
            torch.stack([labels.cpu(), torch.tensor(Transition(*zip(*model.get_replays())).action)]).transpose(0, 1))

    accuracy = 100 * correct.float() / total
    
    model.train()
    dqn.train()
    
    path = torch.cat(path, 0)
    
    return accuracy, path


#%%
acc, path = get_accuracy(model, policy_net)


#%%
import matplotlib.pyplot as plt


