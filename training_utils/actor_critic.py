import torch
import torch.nn as nn
import random

from collections import namedtuple, deque
from math import exp

Transition = namedtuple('Transition',
                        ('state', 'action_J1', 'action_J2', 'next_state', 'next_state_value'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def optimize_predictor(memory, predictor_net, optimizer_predictor, BATCH_SIZE):
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)  
    action_J1_batch = torch.cat(batch.action_J1)
    action_J2_batch = torch.cat(batch.action_J2)
    
    real_next_state = torch.cat(batch.next_state)[:, 20:]

    predicted_next_state = predictor_net(state_batch, action_J1_batch, action_J2_batch)
    # Compute loss
    criterion = nn.MSELoss()
    loss_predictor = criterion(real_next_state, predicted_next_state)

    # Optimize the model
    optimizer_predictor.zero_grad()
    loss_predictor.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(predictor_net.parameters(), 100)
    optimizer_predictor.step()   

    return loss_predictor.clone().detach()

def optimize_critic(memory, critic_net, critic_smooth_net, optimizer_critic, GAMMA = 0.95, BATCH_SIZE = 512):
    if len(memory) < BATCH_SIZE:
        return 0
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    next_state_batch = torch.cat(batch.next_state)
    next_state_hard_critic_batch = torch.cat(batch.next_state_value)

    state_action_values = critic_net(state_batch)
    expected_state_action_values = GAMMA*critic_smooth_net(next_state_batch).squeeze() + (1-GAMMA)*next_state_hard_critic_batch
    
    criterion = nn.MSELoss()
    loss_critic = criterion(state_action_values.squeeze(), expected_state_action_values.squeeze())

    optimizer_critic.zero_grad()
    loss_critic.backward()

    torch.nn.utils.clip_grad_value_(critic_net.parameters(), 100)
    optimizer_critic.step()
    return loss_critic.clone().detach()


def optimize_actor(memory, actor_net, critic_net, optimizer_actor, predictor_net, BATCH_SIZE = 512):
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)

    action_j1 = actor_net(state_batch)
    action_j2 = torch.cat(batch.action_J2)
    #### Actor optimization
    
    next_state_evaluation = torch.cat((state_batch[:, :20], predictor_net(state_batch, action_j1, action_j2)), dim=1)
    q_value = critic_net(next_state_evaluation)
    # Compute loss
    loss_actor = -q_value.mean()

    # Optimize the model
    optimizer_actor.zero_grad()
    loss_actor.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(actor_net.parameters(), 100)
    optimizer_actor.step()   
    return loss_actor.clone().detach()

def select_action(state, actor_net, choose_random_action, device, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return actor_net(state).squeeze()
    else:
        return torch.tensor(choose_random_action(), device=device, dtype=torch.long)
    
def soft_update(policy_net, target_net, TAU=0.005):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)
