import torch
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import save_agent, plot_results

def train_agent(env, max_steps, agent, save_agent_b, path, accumulate_return_steps):

   r_epi, r_frame,= np.zeros(env.num_envs),np.zeros(env.num_envs)
   r_epi_l, r_frame_l = [0], [0]
   state, info = env.reset()  
   step_epi = 0
   epi = 0
   for step in tqdm(range(max_steps)):
      
      action = agent.pi(state,step)
      new_state, reward, done, trunc, info = env.step(action)

      state_trunc = np.copy(state)
      if np.any(trunc):
         if env.num_envs == 1:
            state_trunc[0] = info['final_observation'][0]
         else:
            state_trunc[info['_final_observation']] = info['final_observation'][info['_final_observation']][0]


      agent.train(reward,state_trunc,action,new_state,done,step)
      
      step_epi += 1
      if step%(accumulate_return_steps//env.num_envs) == 0:
         r_frame_l.append(np.mean(r_frame)/(accumulate_return_steps//env.num_envs))
         r_frame= np.zeros(env.num_envs)
   
      if np.any(done == True) or np.any(trunc == True):
         epi += 1
         step_epi =0
         a = np.logical_or(done,trunc)
         if len(r_epi[a]) > 0:
            r_epi_l.extend(r_epi[a].tolist())
         else:
           r_epi_l.append(r_epi[a])
         r_epi[a]=0

      r_epi+=reward
      r_frame+=reward

      state = new_state
   
   if save_agent_b:
      save_agent(agent,path)
   plot_results(r_frame_l,r_epi_l,path,env.num_envs,accumulate_return_steps)










