# pip install ale-py==0.7.4
import argparse
import time

from sb3_contrib import QRDQN, GoExplore
from stable_baselines3 import DDPG, DQN, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import wandb
from experiments.utils import (
    AtariNumberCellsLogger,
    AtariWrapper,
    MaxRewardLogger,
    NumberCellsLogger,
    is_atari,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", type=str, required=True, choices=["dqn", "sac", "ddpg", "td3", "c51", "qr-dqn"], help="Algorithm ID"
    )
    parser.add_argument("--env", type=str, required=True, help="Environment id")
    parser.add_argument("--num-timesteps", type=int, default=50_000_000, help="Number of timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of environments")
    parser.add_argument(
        "--learning-starts", type=int, default=1_000_000, help="Number of random interactions before learning starts"
    )
    parser.add_argument(
        "--nb-random-exploration-steps",
        type=int,
        default=1000,
        help="Number of random exploration steps once the last goal is reached.",
    )
    parser.add_argument("--vec-env-cls", type=str, choices=["subproc", "dummy"], help="Vector environment class")
    parser.add_argument("--tags", type=str, default="", nargs="+", help="List of tags, e.g.: --tag before-modif pr-32")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    algo_id = args.algo
    algo = {"dqn": DQN, "sac": SAC, "ddpg": DDPG, "td3": TD3, "c51": DQN, "qr-dqn": QRDQN}[algo_id]
    policy_kwargs = dict(categorical=True) if args.algo == "c51" else None
    env_id = args.env
    num_timesteps = args.num_timesteps
    n_envs = args.n_envs
    learning_starts = args.learning_starts
    nb_random_exploration_steps = args.nb_random_exploration_steps
    vec_env_cls = {"subproc": SubprocVecEnv, "dummy": DummyVecEnv}.get(args.vec_env_cls)

    run = wandb.init(
        name=f"go-explore__{env_id}__{str(time.time())[-4:]}",
        project="lge",
        config=dict(
            algo=algo_id,
            env_id=env_id,
            num_timesteps=num_timesteps,
            n_envs=n_envs,
            learning_starts=learning_starts,
            nb_random_exploration_steps=nb_random_exploration_steps,
        ),
        sync_tensorboard=True,
        tags=args.tags,
    )

    model_kwargs = dict(policy_kwargs=policy_kwargs)
    env_kwargs = dict()
    if is_atari(env_id):
        model_kwargs["buffer_size"] = 100_000 * n_envs
        env_kwargs["repeat_action_probability"] = 0.25  # Sticky action, needed for v4
    if algo in {DQN, QRDQN}:
        # Take random actions during the `learning_starts` timesteps, then take random
        # actions with decreasing probability during more `learning_starts` timesteps,
        # with a decreasing rate starting at 0.5.
        model_kwargs["exploration_fraction"] = learning_starts / num_timesteps * 2

    model = GoExplore(
        algo,
        env_id,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        learning_starts=learning_starts,
        model_kwargs=model_kwargs,
        wrapper_cls=AtariWrapper if is_atari(env_id) else None,
        nb_random_exploration_steps=nb_random_exploration_steps,
        vec_env_cls=vec_env_cls,
        tensorboard_log=f"runs/{run.id}",
        verbose=1,
    )

    freq = min(int(num_timesteps / 1_000), 100_000)
    callbacks = [MaxRewardLogger(freq)]
    if is_atari(env_id):
        callbacks.append(AtariNumberCellsLogger(freq))
    else:
        callbacks.append(NumberCellsLogger(freq))

    model.explore(num_timesteps, callback=CallbackList(callbacks))
    run.finish()
