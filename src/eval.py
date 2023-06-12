from tqdm import tqdm
from builders import build_eval_environment


def evaluate_agent(agent,args):
    env = build_eval_environment(args)
    state, info = env.reset()
    env.render()
    for _ in range(100):
        state, info = env.reset()
        for step in tqdm(range(1000)):       
            action = agent.pi(state,step)
            new_state, reward, done, trunc, info = env.step(action.item())
            state = new_state
            if done or trunc:
                break
