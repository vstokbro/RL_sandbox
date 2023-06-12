from back_bones import BaseModel, NormDistNet
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class PPO(nn.Module):
    def __init__(self, net, alpha=1, lr=0.001, epsilon=0.1, s_weight=0.01):
        super(PPO, self).__init__()
        self.epsilon = epsilon
        self.net = net
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, maximize=True)
        self.s_weight = s_weight
        self.alpha = alpha

    def forward(self,x):
        return NotImplementedError
    
    def fit(self, advantage, states, actions, log_probs_old):
        self.optimizer.zero_grad()
        m = self.forward(states)
        log_probs = m.log_prob(actions.squeeze())
        if log_probs.ndim == 1:
            log_probs=log_probs.unsqueeze(1)
        epsilon = self.epsilon*self.alpha
        S = m.entropy().mean()
        r = (log_probs-log_probs_old).exp()
        a1 = r*advantage
        a2 =  torch.clamp(r, 1 -epsilon, 1+epsilon)*advantage
        ppo_loss = torch.min(a1,a2).mean()
        loss = ppo_loss + self.s_weight*S   
        loss.backward()
        self.optimizer.step()

class PPO_discrete(PPO):
    def __init__(self, input_featues, output_features, hidden_units, alpha=1, 
                 lr=0.001, epsilon=0.1, s_weight=0.01):
        net = BaseModel(input_featues, output_features, hidden_units)
        super(PPO_discrete, self).__init__(net,alpha,lr,epsilon,s_weight)
        self.net = net

    def forward(self, x):
        output = self.net.forward(x)
        probs =  F.softmax(output, dim=1)
        return Categorical(probs)



class PPO_continous(PPO):
    def __init__(self, input_featues, output_features, hidden_units, 
                 alpha=0.01, lr=0.001, epsilon=0.1, s_weight=0.01, std_grad=True,std_init=0.5):
        net = NormDistNet(input_featues, output_features, hidden_units,std_grad,std_init)
        super(PPO_continous, self).__init__(net,alpha,lr,epsilon,s_weight)
        self.net = net
    
    def forward(self,x):
        means, stds = self.net.forward(x)
        return Normal(means, torch.exp(stds))


class PolicyGradient(nn.Module):
    def __init__(self, net, lr):
        super(PolicyGradient, self).__init__()
        self.net = net
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr,maximize=True)

    def forward(self, x):
        return NotImplementedError

    def fit(self,advantages,states,actions,log_probs_old=1,gammas=1): 
        self.optimizer.zero_grad() 
        pi = self.forward(states)
        log_probs = pi.log_prob(actions.squeeze())
        if log_probs.ndim == 2:
            gammas, advantages = gammas.unsqueeze(1), advantages.unsqueeze(1)
        loss = (gammas*advantages*log_probs).mean()
        loss.backward()
        self.optimizer.step()

class PolicyGradientDiscrete(PolicyGradient):
    def __init__(self, input_featues, output_features, hidden_units,lr):
        net = BaseModel(input_featues,output_features,hidden_units)
        super(PolicyGradientDiscrete, self).__init__(net,lr)

    def forward(self, x):
        output = self.net.forward(x)
        probs =  F.softmax(output, dim=1)
        return Categorical(probs)
    

class PolicyGradientContinous(PolicyGradient):
    def __init__(self, input_featues, output_features, hidden_units,lr,std_grad=True,std_init=0.5):
        net = NormDistNet(input_featues, output_features, hidden_units,std_grad,std_init)
        super(PolicyGradientContinous, self).__init__(net,lr)
    
    def forward(self,x):
        means, stds = self.net.forward(x)
        return Normal(means, torch.exp(stds))



class DQN(nn.Module):
    def __init__(self, input_featues, output_features, hidden_units,gamma,lr):
        super(DQN,self).__init__()
        self.gamma = gamma
        self.V = BaseModel(input_featues,output_features,hidden_units)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
    
    def fit(self,output,targets):  
        self.optimizer.zero_grad() 
        loss = self.loss_fn(output,targets)
        loss.backward()
        self.optimizer.step()
    
    def forward(self,x):
        return self.V(x)


class DDQN(nn.Module):
    def __init__(self, input_featues, output_features, hidden_units,gamma,lr):
        super(DDQN,self).__init__()
        self.gamma = gamma
        self.loss_fn = torch.nn.MSELoss()
        self.A = BaseModel(input_featues,output_features,hidden_units)
        self.V = BaseModel(input_featues,1,hidden_units)
        self.V.fc1 = self.A.fc1
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        
    def fit(self,output,targets):  
        self.optimizer.zero_grad() 
        loss = self.loss_fn(output,targets)
        loss.backward()
        self.optimizer.step()
    
    def forward(self,x):
        return self.V(x)+(self.A(x)-self.A(x).mean())
    