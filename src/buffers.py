from collections import deque
import random
import torch
import numpy as np
from utils import remove_agent_dim



class BatchLoader(object):
    def __init__(self,keys,max_len=100000) -> None:
        self.container = {key:[] for key in keys}
        self.max_len = max_len

    def push(self,transition):
        for i,key in enumerate(self.container.keys()):
            if len(self.container[key])==self.max_len:
                del self.container[key][0]
            self.container[key].append(transition[i].tolist())
    
    def sample(self,batch_size):
        idx = np.random.randint(0,self.__len__(),batch_size)
        batch = [[],[],[],[],[]]
        for i,key in enumerate(self.container.keys()):
            for id in idx:
                batch[i].append(self.container[key][id])
        batch = [torch.tensor(val) for val in batch]
        batch = remove_agent_dim(batch)

        return batch
                
    def get_batch(self,batch_size,batch_n):
        batch = []
        start_batch = batch_n*batch_size
        batch.append(self.container['state'][start_batch:start_batch+batch_size])
        batch.append(self.container['action'][start_batch:start_batch+batch_size])
        batch.append(self.container['log_prob'][start_batch:start_batch+batch_size])
        return map(lambda x: torch.tensor(x), batch)

    def get_all(self):
        return map(lambda x: torch.tensor(x), self.container.values())
    
    def reset(self):
        self.container = {key:[] for key in self.container.keys()}
    def __len__(self):
        return len(self.container['reward'])
