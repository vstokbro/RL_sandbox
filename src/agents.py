import numpy as np
import torch
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from  policy_networks import DQN
from critic_networks import Critic
from buffers import BatchLoader
from utils import remove_agent_dim
from copy import deepcopy



class ActorCritic(object):
    
    def __init__(self,policy_network,gamma,env,input_space, hidden_units, lr, 
               lambda_, n_epochs,batch_size,T) -> None:
        self.gamma=gamma
        self.lambda_= lambda_
        self.batch_size = batch_size
        self.T=T
        self.n_epochs = n_epochs
        self.Pi= policy_network
        self.V = Critic(input_space,1,hidden_units,lr=lr)
        self.batch_loader = BatchLoader(['reward','state','action','new_state,',
                          'done','log_prob'])
        self.n_envs = env.num_envs

    def pi(self,s,steps):
        m = self.Pi(torch.tensor(s))
        action = m.sample().numpy()
        return action
    
    def train(self,reward,state,action,new_state,done,global_step):
        log_prob =  self.Pi(torch.tensor(state)).log_prob(torch.tensor(action)).detach()
        self.batch_loader.push((reward,state,action,new_state,done,log_prob))
        if ((global_step+1) % self.T == 0):
            full_trajectory=self.batch_loader.get_all()
            advantage,val_target = self.gae(full_trajectory)
            for _ in range(self.n_epochs):
                batch_n = 0
                for _ in range(self.T//self.batch_size): 
                    state, action, log_prob = self.batch_loader.get_batch(self.batch_size,batch_n)
                    start_batch = batch_n*self.batch_size
                    advantage_batch=advantage[start_batch:start_batch+self.batch_size]
                    val_target_batch=val_target[start_batch:start_batch+self.batch_size]
                    self.step((state, action, log_prob, advantage_batch,val_target_batch))
                    batch_n+=1
            self.batch_loader.reset()

    def step(self,batch):
        state, action, log_prob, advantage, val_target = remove_agent_dim(list(batch))
        delta = val_target-self.V(state)
        self.Pi.fit(advantage,state,action,log_prob)
        self.V.fit(delta)
        
    def gae(self,batch):
        
        """
        Get generalized advantage estimate of a trajectory
        gamma: trajectory discount (scalar)
        lamda: exponential mean discount (scalar)
        value_old_state: value function result with old_state input
        value_new_state: value function result with new_state input
        reward: agent reward of taking actions in the environment
        done: flag for end of episode
        """
        reward, state, _ ,new_state,done,_ = batch
        value_old_state, value_new_state = self.V(state).reshape(reward.shape), self.V(new_state).reshape(reward.shape) 
        advantage = torch.zeros((reward.shape[0]+1,reward.shape[1]))
        for t in reversed(range(self.T)):
                delta = reward[t] + (self.gamma * value_new_state[t] * ~done[t]) - value_old_state[t]
                advantage[t] = delta + (self.gamma * self.lambda_ * advantage[t + 1] * ~done[t])
        
        advantage = advantage[:self.T]
        value_target = advantage + value_old_state


        return advantage.detach(), value_target.detach()
    
    def return_network(self):
        return self.Pi
    
    def load_from_state_dict(self,path):
        self.Pi = torch.load(path)
    

class Reinforce(object):
    def __init__(self,policy_network,gamma,env,hidden_units, lr,input_space) -> None:
        self.gamma=gamma
        self.Pi= policy_network
        self.baseline = Critic(input_space,1,hidden_units,lr=lr)
        self.batch_loader = BatchLoader(['reward','state','action'])

    
    def pi(self,s,steps):
        m = self.Pi(torch.tensor(s))
        action = m.sample().numpy()
        return action
        
    def train(self,reward,state,action,new_state,done,global_step):
        self.batch_loader.push((reward,state,action))
        if done:
            batch = self.batch_loader.get_all()
            rewards, states, actions = batch
            n_steps = len(rewards)
            G = self.calculate_returns(rewards)
            targets = (G-self.baseline(states))
            gammas = (self.gamma**torch.arange(n_steps)).unsqueeze(1)
            self.Pi.fit(targets.detach(),states,actions,gammas=gammas)
            self.baseline.fit(targets)
            self.batch_loader.reset()
    
    def calculate_returns(self, rewards):
        max_steps = len(rewards)
        returns = torch.zeros(max_steps+1)
        for t in reversed(range(max_steps)):
            returns[t] = rewards[t]+self.gamma*returns[t+1]
        return returns[:max_steps].unsqueeze(1)
    
    def return_network(self):
        return self.Pi
    
    def load_from_state_dict(self,path):
        self.Pi = torch.load(path)
    

    
    
    
class DeepQ(object):
    
    def __init__(self,env,hidden_units,gamma,batch_size,n_batches,
                 max_buffer_len,delay, epsilon,lr,
                 steps_b_rpl,C) -> None:
        self.n_actions = env.action_space[0].n
        self.input_space = env.observation_space.shape[1] 
        self.Q = DQN(self.input_space,self.n_actions,hidden_units,gamma,lr)
        self.Q_target = DQN(self.input_space,self.n_actions,hidden_units,gamma,lr)
        
        self.buffer = BatchLoader(['reward','state','action','new_state,',
                          'done'],max_buffer_len)
        self.delay =delay
        self.gamma = gamma
        self.batch_size = batch_size 
        self.n_batches = n_batches
        self.epsilon = epsilon
        self.step_b_rpl = steps_b_rpl
        self.C = C
        self.num_envs = env.num_envs

    
    def pi(self,s,steps):
        if callable(self.epsilon):
            epsilon = self.epsilon(steps)
        else: 
            epsilon = self.epsilon
        if random.random() < epsilon:
            a = np.random.randint(0,self.n_actions-1,(self.num_envs))
        else: 
            a = torch.argmax(self.Q(torch.tensor(s)),axis=1).detach().numpy()
        return a

    def q(self,s,a):
        return self.Q(s)[a]

    def train(self,reward,state,action,new_state,done,global_step):
        self.buffer.push((reward,state,action,new_state,done))
        if global_step % self.step_b_rpl==0:
            self.experience_replay()
        if global_step % self.C == 0:
            self.Q_target = deepcopy(self.Q)

    def experience_replay(self):
        if self.delay<=len(self.buffer):
            for _ in range(self.n_batches):
                    batch = self.buffer.sample(self.batch_size)
                    r,s,a,sn,done = batch
                    r,done,a = r.squeeze(), done.squeeze(), a.squeeze()
                    y = self.Q(s).double()
                    max_idx = torch.argmax(self.Q(sn),1)
                    row_v = torch.arange(self.batch_size*self.num_envs)
                    max_values = self.Q_target(sn)[row_v,max_idx]
                    y_target =  (r+self.gamma*max_values)*(~done)
                    y_a = y[row_v,a]
                    self.Q.fit(y_a,y_target.detach().double())
    
    def return_network(self):
        return self.Q
    
    def load_from_state_dict(self,path):
        self.Q = torch.load(path)
    
