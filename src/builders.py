import argparse
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from agents import ActorCritic, DeepQ, Reinforce
from policy_networks import (
    PPO_continous,
    PPO_discrete,
    PolicyGradientDiscrete,
    PolicyGradientContinous,
)
from utils import linear_interp
import gymnasium as gym


def build_eval_environment(args):
    if args.continous:
        env = gym.make(
            args.env_name,
            continuous=args.continous,
            max_episode_steps=args.max_episode_len,
            render_mode='human'
        )
    else:
            env = gym.make(
            args.env_name,
            max_episode_steps=args.max_episode_len,
            render_mode='human'
        )
    return env

def build_environment(args):
    if args.env_name=="LunarLander-v2":
            env = gym.vector.make(
                args.env_name,
                continuous=args.continous,
                max_episode_steps=args.max_episode_len,
                num_envs=args.num_envs,
            )
    else:
            env = gym.vector.make(
                args.env_name,
                max_episode_steps=args.max_episode_len,
                num_envs=args.num_envs,
            )
    return env



def build_agent(args, env):
    if args.agent == "deepq":
        epsilon = linear_interp(
            args.max_eps, args.min_eps, args.eps_delay, args.min_iter
        )
        agent = DeepQ(
            env=env,
            hidden_units=[args.hidden_units_1, args.hidden_units_2],
            gamma=args.gamma,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            max_buffer_len=args.max_buffer_len,
            delay=args.train_delay,
            epsilon=epsilon,
            lr=args.lr,
            steps_b_rpl=args.steps_b_rpl,
            C=args.steps_b_upd_trg_net,
        )

    else:
        input_space = env.observation_space.shape[1]
        if args.continous:
            action_space_size = env.action_space.shape[1]
            if args.PPO and not (args.agent == "reinforce"):
                policy_network = PPO_continous(
                    input_featues=input_space,
                    output_features=action_space_size,
                    hidden_units=[args.hidden_units_1, args.hidden_units_2],
                    alpha=args.alpha,
                    lr=args.lr,
                    epsilon=args.eps_PPO,
                    s_weight=args.entropy_weight,
                    std_grad=args.std_grad,
                    std_init=args.std_init,
                )
            else:
                policy_network = PolicyGradientContinous(
                    input_featues=input_space,
                    output_features=action_space_size,
                    hidden_units=[args.hidden_units_1, args.hidden_units_2],
                    lr=args.lr,
                    std_grad=args.std_grad,
                    std_init=args.std_init,
                )

        else:
            action_space_size = env.action_space[0].n
            if args.PPO and not (args.agent == "reinforce"):
                policy_network = PPO_discrete(
                    input_featues=input_space,
                    output_features=action_space_size,
                    hidden_units=[args.hidden_units_1, args.hidden_units_2],
                    alpha=args.alpha,
                    lr=args.lr,
                    epsilon=args.eps_PPO,
                    s_weight=args.entropy_weight,
                )
            else:
                policy_network = PolicyGradientDiscrete(
                    input_featues=input_space,
                    output_features=action_space_size,
                    hidden_units=[args.hidden_units_1, args.hidden_units_2],
                    lr=args.lr,
                )

        if args.agent == "actor_critic":
            agent = ActorCritic(
                policy_network,
                args.gamma,
                env,
                input_space,
                [args.hidden_units_1, args.hidden_units_2],
                args.lr,
                args.lambda_,
                args.n_epochs,
                args.batch_size,
                args.update_horizon,
            )

        elif args.agent == "reinforce":
            agent = Reinforce(
                policy_network=policy_network,
                gamma=args.gamma,
                env=env,
                hidden_units=[args.hidden_units_1, args.hidden_units_2],
                lr=args.lr,
                input_space=input_space,
            )

    return agent
