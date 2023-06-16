import argparse


def build_argparser():
    parser = argparse.ArgumentParser(description="run configuration")

    #
    parser.add_argument(
        "-run_name", default="NO_NAME", help="save name of the run"
    )
    parser.add_argument(
        "-eval",
        default=0,
        type=int,
        help="0 to train model, 1 to evaluate model",
    )
    parser.add_argument(
        "-eval_path", default="", type=str, help="path to model.ckpt to evaluate"
    )

    # Trainer parameters
    parser.add_argument(
        "-save_policy",
        default=True,
        type=int,
        help="save policy network at the end of training",
    )
    parser.add_argument(
        "-max_iterations",
        default=100000,
        type=int,
        help="max number of iterations to run for training",
    )
    parser.add_argument(
        "-accumulate_return_steps",
        default=10000,
        type=int,
        help="number of steps to accumulate return for diagnostics plots",
    )

    # Env parameters
    parser.add_argument(
        "-env_name",
        default="LunarLander-v2",
        choices=["LunarLander-v2", "CartPole-v1"],
        help="environment to train on",
    )

    parser.add_argument(
        "-num_envs",
        default=1,
        type=int,
        help="number of environments to train in parallel",
    )
    parser.add_argument(
        "-max_episode_len",
        default=1000,
        type=int,
        help="max number of steps in an episode before termination",
    )

    # Shared agent parameters
    parser.add_argument(
        "-agent",
        default="actor_critic",
        choices=["actor_critic", "deepq", "reinforce"],
        help="type of agent to train or evaluate",
    )
    parser.add_argument(
        "-batch_size",
        default=4,
        type=int,
        help="batch size per agent for training (if you triain 10 agents in parallel, batch size is 4, then 40 samples are used for training)",
    )
    parser.add_argument("-lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument(
        "-gamma", default=0.99, type=float, help="discount factor"
    )
    parser.add_argument(
        "-hidden_units_1",
        default=512,
        type=int,
        help="number of hidden units in first layer",
    )
    parser.add_argument(
        "-hidden_units_2",
        default=128,
        type=int,
        help="number of hidden units in second layer",
    )

    # Actor critic parameters
    parser.add_argument(
        "-update_horizon",
        default=32,
        type=int,
        help="number of steps before updating policy network",
    )
    parser.add_argument(
        "-n_epochs", default=4, type=int, help="number of epochs for each update"
    )
    parser.add_argument(
        "-alpha",
        default=1,
        type=int,
        help="alpha parameter to reweight entropy loss",
    )
    parser.add_argument(
        "-std_grad",
        default=1,
        type=int,
        help="1 to use make std trainable, 0 to use fixed std",
    )
    parser.add_argument(
        "-std_init", default=0.5, type=float, help="initial value of std"
    )
    parser.add_argument(
        "-continous",
        default=0,
        type=int,
        help="1 for continous action space, 0 for discrete action space",
    )
    parser.add_argument(
        "-lambda_",
        default=0.95,
        type=float,
        help="exponential decay rate for generalized advantage estimates",
    )

    # PPO parameters
    parser.add_argument(
        "-PPO",
        default=1,
        type=float,
        help="1 to use PPO loss, 0 to use vanilla actor critic",
    )
    parser.add_argument(
        "-entropy_weight",
        default=0.01,
        type=float,
        help="weight for entropy loss",
    )
    parser.add_argument(
        "-eps_PPO",
        default=0.1,
        type=float,
        help="clipping parameter for PPO loss",
    )

    # DQN parameters
    parser.add_argument(
        "-max_buffer_len",
        default=100000,
        type=int,
        help="max number of samples in replay buffer",
    )
    parser.add_argument("-train_delay", default=1000, type=int)
    parser.add_argument(
        "-eps_delay",
        default=1000,
        type=int,
        help="steps before epsilon starts decreasing",
    )
    parser.add_argument(
        "-min_eps", default=0.1, type=float, help="minimum value of epsilon"
    )
    parser.add_argument(
        "-max_eps", default=0.9, type=float, help="maximum value of epsilon"
    )
    parser.add_argument(
        "-min_iter",
        default=50000,
        type=float,
        help="number of iterations before epsilon reaches min_eps",
    )
    parser.add_argument(
        "-steps_b_rpl", default=10, type=int, help="steps before replay"
    )
    parser.add_argument(
        "-steps_b_upd_trg_net",
        default=50,
        type=int,
        help="steps before updating target network",
    )
    parser.add_argument(
        "-n_batches",
        default=10,
        type=int,
        help="number of batches to sample from replay buffer for each update",
    )

    return parser
