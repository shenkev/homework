import sys
import gym.spaces
import itertools
import random
import numpy as np
import tensorflow                as tf
import torch
import torch.nn as nn
import math
from tqdm import tqdm

sys.path.append("./../")
from logger import Logger

from collections import namedtuple
from dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


# def atari_model(img_in, num_actions, scope, reuse=False):
#     # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
#     with tf.variable_scope(scope, reuse=reuse):
#         out = img_in
#         with tf.variable_scope("convnet"):
#             # original architecture
#             out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
#             out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
#             out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
#         out = layers.flatten(out)
#         with tf.variable_scope("action_value"):
#             out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
#             out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
#
#         return out

def conv_block(in_chan, out_chan, k_size, stride, pad, opts):
    block = nn.Sequential()
    block.add_module('conv_1', nn.Conv2d(in_chan, out_chan, kernel_size=k_size, stride=stride, padding=pad))

    if opts['bn']:
        block.add_module('bn_1', nn.BatchNorm2d(out_chan))

    block.add_module('relu_1', nn.ReLU())

    if opts['mp']:
        block.add_module('mp_1', nn.MaxPool2d(kernel_size=2, stride=2))

    return block


class Q_model(nn.Module):

    def __init__(self, input_shape, output_size, args):

        super(Q_model, self).__init__()

        self.height, self.width, self.chans = input_shape
        self.output_size = output_size
        self.opts = {
            'bn': args.batch_norm,
            'mp': args.max_pool
        }

        self.layer1 = conv_block(self.chans, 32, 8, 4, 2, self.opts)
        self.layer2 = conv_block(32, 64, 4, 2, 2, self.opts)
        self.layer3 = conv_block(64, 64, 3, 1, 1, self.opts)

        final_width = math.ceil(self.height/math.ceil(4*2*1))
        self.fc1 = nn.Linear(final_width*final_width*64, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, self.output_size)

    def forward(self, x):

        z = self.layer1(x)
        z = self.layer2(z)
        z = self.layer3(z)
        z = z.reshape(z.size(0), -1)
        z = self.fc1(z)
        z = self.relu1(z)
        out = self.fc2(z)
        return out


def learn(env,
          args,
          lr_schedule,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    logger = Logger('./logs/' + 'default')

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    Q_net = Q_model(input_shape, num_actions, args)
    target_net = Q_model(input_shape, num_actions, args)

    if torch.cuda.is_available():
        Q_net.cuda()
        target_net.cuda()

    target_net.load_state_dict(Q_net.state_dict())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Q_net.parameters()), lr=lr_schedule.value(0), eps=1e-4)
    loss_fnc = torch.nn.SmoothL1Loss()

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    ######

    # # update_target_fn will be called periodically to copy Q network to target Q network
    # update_target_fn = []
    # for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
    #                            sorted(target_q_func_vars, key=lambda v: v.name)):
    #     update_target_fn.append(var_target.assign(var))
    # update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    pbar = tqdm(total=10000000)

    for t in itertools.count():
        if t%100==0:
            pbar.update(100)

        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)

        #####

        idx = replay_buffer.store_frame(last_obs)

        if random.random() < exploration.value(t):
            action = random.randint(0, num_actions-1)
        else:
            state = replay_buffer.encode_recent_observation()
            state = to_var(torch.from_numpy(state).float()/255.0).permute(2, 0, 1).unsqueeze(0)
            actions = Q_net(state)
            q_value, q_idx = torch.max(actions, dim=1)
            action = to_np(q_idx)[0]

        obs, reward, done, info = env.step(action)
        replay_buffer.store_effect(idx, action, reward, done)

        if done:
            last_obs = env.reset()
        else:
            last_obs = obs

        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # 3.b: initialize the model if it has not been initialized yet; to do
            # that, call
            #    initialize_interdependent_variables(session, tf.global_variables(), {
            #        obs_t_ph: obs_t_batch,
            #        obs_tp1_ph: obs_tp1_batch,
            #    })
            # where obs_t_batch and obs_tp1_batch are the batches of observations at
            # the current and next time step. The boolean variable model_initialized
            # indicates whether or not the model has been initialized.
            # Remember that you have to update the target network too (see 3.d)!
            # 3.c: train the model. To do this, you'll need to use the train_fn and
            # total_error ops that were created earlier: total_error is what you
            # created to compute the total Bellman error in a batch, and train_fn
            # will actually perform a gradient step and update the network parameters
            # to reduce total_error. When calling session.run on these you'll need to
            # populate the following placeholders:
            # obs_t_ph
            # act_t_ph
            # rew_t_ph
            # obs_tp1_ph
            # done_mask_ph
            # (this is needed for computing total_error)
            # learning_rate -- you can get this from optimizer_spec.lr_schedule.value(t)
            # (this is needed by the optimizer to choose the learning rate)
            # 3.d: periodically update the target network by calling
            # session.run(update_target_fn)
            # you should update every target_update_freq steps, and you may find the
            # variable num_param_updates useful for this (it was initialized to 0)
            #####

            # 3.a sample a batch of transitions
            obs, act, rew, next_obs, done_mask = replay_buffer.sample(batch_size)
            obs, next_obs = [(torch.from_numpy(x).float()/255.0).permute(0, 3, 1, 2) for x in [obs, next_obs]]
            act, rew, done_mask = [torch.from_numpy(x) for x in [act, rew, done_mask]]
            obs, next_obs, act, rew, done_mask = [to_var(x) for x in [obs, next_obs, act, rew, done_mask]]

            # 3.c train the model
            set_lr(optimizer, lr_schedule.value(t))
            optimizer.zero_grad()

            pred_q = Q_net(obs)
            pred_q_a = pred_q.gather(1, act.long().unsqueeze(1)).squeeze(1)

            target_q = target_net(next_obs)
            if args.doubleQ:
                _, argmax_q = torch.max(pred_q, dim=1)
                target_q_a = target_q.gather(1, argmax_q.unsqueeze(1)).squeeze(1)
            else:
                target_q_a, _ = torch.max(target_q, dim=1)
            target = rew + (1-done_mask)*gamma*target_q_a
            target = target.detach()

            loss = loss_fnc(pred_q_a, target)
            loss.backward()
            gradient_noise_and_clip(Q_net.parameters(), grad_norm_clipping)
            optimizer.step()
            num_param_updates += 1

            # 3.d update target network
            if num_param_updates % target_update_freq == 0:
                # copy over weights from Q-net to target-net
                target_net.load_state_dict(Q_net.state_dict())

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            logger.scalar_summary('mean_episode_reward', mean_episode_reward, t)
            logger.scalar_summary('best_mean_episode_reward', best_mean_episode_reward, t)
            logger.scalar_summary('episodes', len(episode_rewards), t)
            logger.scalar_summary('exploration', exploration.value(t), t)
            logger.scalar_summary('learning_rate', lr_schedule.value(t), t)
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % lr_schedule.value(t))
            sys.stdout.flush()

    pbar.close()
