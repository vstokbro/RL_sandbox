from tqdm import tqdm
from builders import build_eval_environment


def evaluate_agent(agent,args):
    env = build_eval_environment(args)
    if args.agent=="deepq":
        agent.epsilon=0
    state, info = env.reset()
    env.render()
    for _ in range(100):
        state, info = env.reset()
        for step in tqdm(range(1000)):       
            action = agent.pi(state,step)
            if not args.continous:
                action = action.item()
            else: 
                action = action.squeeze()
            new_state, reward, done, trunc, info = env.step(action)
            state = new_state
            if done or trunc:
                break
