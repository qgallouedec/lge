# # pip install ale-py==0.7.4
import argparse
import time

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import wandb
from experiments.utils import AtariNumberCellsLogger, AtariWrapper, MaxRewardLogger, MyBuffer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment id")
    parser.add_argument("--num-timesteps", type=int, default=10_000_000, help="Number of timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of environments")
    parser.add_argument(
        "--learning-starts", type=int, default=1_000_000, help="Number of random interactions before learning starts"
    )
    parser.add_argument("--vec-env-cls", type=str, choices=["subproc", "dummy"], help="Vector environment class")
    parser.add_argument("--tags", type=str, default=[], nargs="+", help="List of tags, e.g.: --tag before-modif pr-32")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env_id = args.env
    num_timesteps = args.num_timesteps
    n_envs = args.n_envs
    learning_starts = args.learning_starts
    vec_env_cls = {"subproc": SubprocVecEnv, "dummy": DummyVecEnv}.get(args.vec_env_cls)

    run = wandb.init(
        name=f"c51__{env_id}__{str(time.time())[-4:]}",
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

    venv = make_vec_env(
        env_id,
        n_envs=n_envs,
        wrapper_class=AtariWrapper,
        env_kwargs={"repeat_action_probability": 0.25},
        vec_env_cls=vec_env_cls,
    )
    venv = VecMonitor(venv)

    model = DQN(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(categorical=True),
        buffer_size=100_000 * n_envs,
        learning_starts=learning_starts,
        replay_buffer_class=MyBuffer,
        exploration_fraction=learning_starts / num_timesteps * 2,
        tensorboard_log=f"runs/{run.id}",
        verbose=1,
    )

    freq = min(int(num_timesteps / 1000), 100_000)
    number_cells_logger = AtariNumberCellsLogger(freq)
    max_reward_logger = MaxRewardLogger(freq)
    model.learn(num_timesteps, callback=CallbackList([number_cells_logger, max_reward_logger]))
    run.finish()

1000 > 150000000 /100000