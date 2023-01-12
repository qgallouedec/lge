# pip install ale-py==0.7.4
import argparse
import time

from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import wandb
from experiments.utils import (
    AtariNumberCellsLogger,
    AtariWrapper,
    MaxRewardLogger,
    NumberCellsLogger,
    is_atari,
    ReplayBufferWithInfo,
)
from stable_baselines3.common.env_util import make_vec_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment id")
    parser.add_argument("--num-timesteps", type=int, default=100_000_000, help="Number of timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of environments")
    parser.add_argument(
        "--learning-starts", type=int, default=1_000_000, help="Number of random interactions before learning starts"
    )
    parser.add_argument("--vec-env-cls", type=str, choices=["subproc", "dummy"], help="Vector environment class")
    parser.add_argument("--tags", type=str, default="", nargs="+", help="List of tags, e.g.: --tag before-modif pr-32")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env_id = args.env
    num_timesteps = args.num_timesteps
    n_envs = args.n_envs
    learning_starts = args.learning_starts
    vec_env_cls = {"subproc": SubprocVecEnv, "dummy": DummyVecEnv}.get(args.vec_env_cls)

    run = wandb.init(
        name=f"qrdqn__{env_id}__{str(time.time())[-4:]}",
        project="lge",
        config=dict(
            env_id=env_id,
            num_timesteps=num_timesteps,
            n_envs=n_envs,
            learning_starts=learning_starts,
        ),
        sync_tensorboard=True,
        tags=args.tags,
    )

    env_kwargs = dict()
    if is_atari(env_id):
        env_kwargs["repeat_action_probability"] = 0.25  # Sticky action, needed for v4
    # Take random actions during the `learning_starts` timesteps, then take random
    # actions with decreasing probability during more `learning_starts` timesteps,
    # with a decreasing rate starting at 0.5.
    wrapper_cls = AtariWrapper if is_atari(env_id) else None
    venv = make_vec_env(env_id, n_envs=n_envs, wrapper_class=wrapper_cls, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls)

    model = QRDQN(
        "CnnPolicy",
        venv,
        buffer_size=100_000 * n_envs,
        learning_starts=learning_starts,
        train_freq=10,
        replay_buffer_class=ReplayBufferWithInfo,
        exploration_fraction=learning_starts / num_timesteps * 2,
        tensorboard_log=f"runs/{run.id}",
        verbose=1,
    )

    freq = min(int(num_timesteps / 1_000), 100_000)
    callbacks = [MaxRewardLogger(freq)]
    if is_atari(env_id):
        callbacks.append(AtariNumberCellsLogger(freq))
    else:
        callbacks.append(NumberCellsLogger(freq))

    model.learn(num_timesteps, callback=CallbackList(callbacks), log_interval=1000)
    run.finish()
