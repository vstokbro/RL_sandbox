from back_bones import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F



class Critic(nn.Module):
    def __init__(self, input_featues, output_features, hidden_units,lr):
        super(Critic, self).__init__()
        self.V = BaseModel(input_featues, output_features, hidden_units)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.loss_fn = nn.MSELoss()
        self.loss_list = []


    def fit(self,targets): 
        self.optimizer.zero_grad() 
        loss = targets.pow(2).mean()
        loss.backward()
        self.optimizer.step()
        self.loss_list.append(loss.item())
    
    def forward(self,x):
        return self.V(x)
        
