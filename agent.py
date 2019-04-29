import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

BATCH_SIZE = 128
DISCOUNT_RATE = 0.99
TAU = 0.01
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3

from model import ValueNetwork, PolicyNetwork

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class Agent():
    def __init__(self, state_size, action_size, num_agents):
        state_dim  = state_size
        #agent_input_state_dim = state_size*2 # Previos state is passed in with with the current state.
        action_dim = action_size
        
        self.num_agents = num_agents
        
        max_size = 100000 ###
        self.replay = Replay(max_size)
        
        hidden_dim = 128
        
        self.critic_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        self.actor_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data)
        
        self.critic_optimizer  = optim.Adam(self.critic_net.parameters(),  lr=CRITIC_LEARNING_RATE)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=ACTOR_LEARNING_RATE)
        
    def get_action(self, state):
        return self.actor_net.get_action(state)[0]
        
    def add_replay(self, state, action, reward, next_state, done):
        for i in range(self.num_agents):
            self.replay.add(state[i], action[i], reward[i], next_state[i], done[i])
        
    def learning_step(self):
        
        #Check if relay buffer contains enough samples for 1 batch
        if (self.replay.cursize < BATCH_SIZE):
            return
        
        #Get Samples
        state, action, reward, next_state, done = self.replay.get(BATCH_SIZE)

        #calculate loss
        actor_loss = self.critic_net(state, self.actor_net(state))
        actor_loss = -actor_loss.mean()

        next_action    = self.target_actor_net(next_state)
        target_value   = self.target_critic_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * DISCOUNT_RATE * target_value

        value = self.critic_net(state, action)
        critic_loss = F.mse_loss(value, expected_value.detach())

        #backprop
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #soft update
        self.soft_update(self.critic_net,  self.target_critic_net,  TAU)
        self.soft_update(self.actor_net, self.target_actor_net, TAU)
        
    def save(self, name):
        torch.save(self.critic_net.state_dict(), name + "_critic")
        torch.save(self.actor_net.state_dict(), name + "_actor")
        
    def load(self, name):
        self.critic_net.load_state_dict(torch.load(name + "_critic"))
        self.critic_net.eval()
        self.actor_net.load_state_dict(torch.load(name + "_actor"))
        self.actor_net.eval()
        
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data)
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            
class Replay:
    def __init__(self,maxsize):
        self.buffer = []
        self.maxsize = maxsize
        self.cursize = 0
        self.indx = 0
        
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if (self.cursize < self.maxsize):
            self.buffer.append(data)
            self.cursize += 1
        else :
            self.buffer[self.indx] =  data
        self.indx += 1
        self.indx = self.indx % self.maxsize
    
    def get(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        state      = torch.FloatTensor(state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        
        return state, action, reward, next_state, done
        
        
