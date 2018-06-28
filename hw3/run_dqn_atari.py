import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import torch

import dqn
from dqn_utils import *
from atari_wrappers import *


def atari_learn(env,
                args,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        args,
        lr_schedule=lr_schedule,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

# def get_available_gpus():
#     from tensorflow.python.client import device_lib
#     local_device_protos = device_lib.list_local_devices()
#     return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

# def set_global_seeds(i):
#     try:
#         import tensorflow as tf
#     except ImportError:
#         pass
#     else:
#         tf.set_random_seed(i)
#     np.random.seed(i)
#     random.seed(i)

def set_global_seeds(i):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)

# def get_session():
#     tf.reset_default_graph()
#     tf_config = tf.ConfigProto(
#         inter_op_parallelism_threads=1,
#         intra_op_parallelism_threads=1)
#     session = tf.Session(config=tf_config)
#     return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = './videos/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--max_pool', action='store_true')

    args = parser.parse_args()

    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = random.randint(0, 1000)
    print("Seed: {}".format(seed))
    env = get_env(task, seed)
    atari_learn(env, args, num_timesteps=task.max_timesteps)

if __name__ == "__main__":
    main()
