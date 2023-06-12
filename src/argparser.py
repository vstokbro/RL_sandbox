import argparse

def build_argparser():
    parser = argparse.ArgumentParser(description="run configuration")

    #
    parser.add_argument("-run_name", default="NO_NAME")
    parser.add_argument("-eval", default=0, type=int)
    parser.add_argument("-eval_path", default="", type=str)

    # Trainer parameters
    parser.add_argument("-save_policy", default=True, type=int)
    parser.add_argument("-max_iterations", default=100000, type=int)
    parser.add_argument("-accumulate_return_steps", default=10000, type=int)

    # Env parameters
    parser.add_argument(
        "-env_name", default="LunarLander-v2", choices=["LunarLander-v2", "CartPole-v1"]
    )
    parser.add_argument("-num_envs", default=1, type=int)
    parser.add_argument("-max_episode_len", default=1000, type=int)

    # Shared agent parameters
    parser.add_argument(
        "-agent", default="actor_critic", choices=["actor_critic", "deepq", "reinforce"]
    )
    parser.add_argument("-batch_size", default=4, type=int)
    parser.add_argument("-lr", default=0.0001, type=float)
    parser.add_argument("-gamma", default=0.99, type=float)
    parser.add_argument("-hidden_units_1", default=512, type=int)
    parser.add_argument("-hidden_units_2", default=128, type=int)

    # Actor critic parameters
    parser.add_argument("-update_horizon", default=32, type=int)
    parser.add_argument("-n_epochs", default=4, type=int)
    parser.add_argument("-alpha", default=1, type=int)
    parser.add_argument("-std_grad", default=1, type=int)
    parser.add_argument("-std_init", default=0.5, type=float)
    parser.add_argument("-continous", default=0, type=int)
    parser.add_argument("-lambda_", default=0.95, type=float)

    # PPO parameters
    parser.add_argument("-PPO", default=1, type=float)
    parser.add_argument("-entropy_weight", default=0.01, type=float)
    parser.add_argument("-eps_PPO", default=0.1, type=float)

    # DQN parameters
    parser.add_argument("-max_buffer_len", default=100000, type=int)
    parser.add_argument("-train_delay", default=1000, type=int)
    parser.add_argument(
        "-eps_delay",
        default=1000,
        type=int,
        help="steps before epsilon starts decreasing",
    )
    parser.add_argument("-min_eps", default=0.1, type=float)
    parser.add_argument("-max_eps", default=0.9, type=float)
    parser.add_argument("-min_iter", default=50000, type=float)
    parser.add_argument("-steps_b_rpl", default=10, type=int)
    parser.add_argument("-steps_b_upd_trg_net", default=50, type=int)
    parser.add_argument("-sum_grad_actions", default=False, type=int)
    parser.add_argument("-n_batches", default=10, type=int)

    return parser