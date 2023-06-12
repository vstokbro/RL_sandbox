import argparse
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import json

def remove_agent_dim(values):
    for i,val in enumerate(values) :
        if val.ndim == 3:
            values[i] = val.reshape(-1,val.shape[-1])
        else:
            values[i] = val.reshape(-1,1)
    return values

def make_dirs(args,overwrites):
    p = Path(f"outputs/{args.agent}/{args.run_name}")
    p.mkdir(parents=True, exist_ok=True)

    with open(p / "args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    with open(p / "args_overwrites.txt", "w") as f:
        f.write("\n".join(overwrites))
    return p


def linear_interp(maxval, minval, delay, miniter):
    """
    Will return a function f(i) with the following signature:

    f(i) = maxval for i < delay
    f(i) = linear interpolate between max/minval until delay+miniter
    f(i) = minval for i > delay+miniter
    """
    return lambda steps: min(max([maxval - ((steps - delay) / miniter) * (maxval - minval), minval]), maxval)

def save_agent(agent,path):
    save_path = os.path.join(path,'model')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file = os.path.join(save_path,'model.ckpt')
    net = agent.return_network()
    torch.save(net,file)

def plot_results(return_frame,return_epi,path,num_envs,accumulate_return_steps):
    save_path = os.path.join(path,'plots')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    plt.plot(range(len(return_epi)),return_epi)
    plt.title('Return per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.grid() 
    plt.savefig(os.path.join(save_path,'return_p_episode.png'))
    plt.cla()
    plt.plot(np.arange(len(return_frame))*accumulate_return_steps,return_frame)
    plt.title('Return per frame')
    plt.xlabel('Frames')
    plt.ylabel('Return')
    plt.grid() 
    plt.savefig(os.path.join(save_path,'return_p_frame.png'))
    plt.cla()
    plt.plot(np.arange(len(return_frame))*accumulate_return_steps//num_envs,return_frame)
    plt.title('Return per iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Return')
    plt.grid() 
    plt.savefig(os.path.join(save_path,'return_p_iteration.png'))
     
     

