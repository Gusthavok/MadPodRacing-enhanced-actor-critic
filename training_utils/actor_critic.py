import torch
import torch.nn as nn
import random

from collections import namedtuple, deque
from math import exp

import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action_J1', 'action_J2', 'next_state', 'state_hard_value'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.weights = list(range(1, capacity+1))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        weights = self.weights[:len(self.memory)]
        return random.choices(self.memory, weights=weights, k=batch_size)

    def __len__(self):
        return len(self.memory)

def optimize_predictor(batch, predictor_net, optimizer_predictor, test_dataset):
    state_batch = torch.cat(batch.state)  
    action_J1_batch = torch.cat(batch.action_J1)
    action_J2_batch = torch.cat(batch.action_J2)
    
    real_next_state = torch.cat(batch.next_state)[:, 20:]

    if test_dataset:
        with torch.no_grad():
            predicted_next_state = predictor_net(state_batch, action_J1_batch, action_J2_batch)
    else:
        predicted_next_state = predictor_net(state_batch, action_J1_batch, action_J2_batch)
    
    # Compute loss
    criterion = nn.MSELoss()
    loss_predictor = criterion(real_next_state, predicted_next_state)

    if not test_dataset:
        # Optimize the model
        optimizer_predictor.zero_grad()
        loss_predictor.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(predictor_net.parameters(), 50)
        optimizer_predictor.step()   

    return loss_predictor.clone().detach()

def optimize_critic(batch, critic_net, critic_smooth_net, optimizer_critic, test_dataset, GAMMA = 0.95):
    state_batch = torch.cat(batch.state)
    next_state_batch = torch.cat(batch.next_state)
    next_state_hard_critic_batch = torch.cat(batch.state_hard_value)

    with torch.no_grad():
        expected_state_action_values = GAMMA*critic_smooth_net(next_state_batch).squeeze() + (1-GAMMA)*next_state_hard_critic_batch
    
    if test_dataset:
        with torch.no_grad():
            state_action_values = critic_net(state_batch)
    else:
        state_action_values = critic_net(state_batch)

    
    criterion = nn.MSELoss()
    loss_critic = criterion(expected_state_action_values.squeeze(), state_action_values.squeeze())

    if not test_dataset:
        optimizer_critic.zero_grad()
        loss_critic.backward()

        torch.nn.utils.clip_grad_value_(critic_net.parameters(), 50)
        optimizer_critic.step()
    return loss_critic.clone().detach()


def optimize_actor(batch, actor_net, critic_smooth_net, optimizer_actor, predictor_net, test_dataset):
    state_batch = torch.cat(batch.state)
    
    if test_dataset:
        with torch.no_grad():
            action_j1 = actor_net(state_batch)
    else:
        action_j1 = actor_net(state_batch)
        
    action_j2 = torch.cat(batch.action_J2)

    
    # Compute loss
    next_state_evaluation = torch.cat((state_batch[:, :20], predictor_net(state_batch, action_j1, action_j2)), dim=1)
    q_value = critic_smooth_net(next_state_evaluation)
    loss_actor = -q_value.mean()

    if not test_dataset:
        # Optimize the model
        optimizer_actor.zero_grad()
        loss_actor.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(actor_net.parameters(), 50)
        optimizer_actor.step()   
    return loss_actor.clone().detach()

def select_action(state, actor_net, choose_random_action, device, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return actor_net(state).squeeze()
    else:
        return torch.tensor(choose_random_action(state), device=device, dtype=torch.long)
    
def soft_update(policy_net, target_net, TAU=0.005):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)
