from builders import build_agent, build_environment
from argparser import build_argparser
from utils import make_dirs
import gymnasium as gym
import os
from pathlib import Path
from train import train_agent
from eval import evaluate_agent
import json
import sys


def main(args):
    env = build_environment(args)

    agent = build_agent(args, env)

    
    if args.eval:
        p = Path(args.eval_path)
        agent.load_from_state_dict(p)
        evaluate_agent(agent,args)

    else:
        p = make_dirs(args,sys.argv[1:])
        train_agent(
            env,
            args.max_iterations,
            agent,
            args.save_policy,
            p,
            args.accumulate_return_steps,
        )


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
