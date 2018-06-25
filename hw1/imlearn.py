import sys
from time import time
from shutil import copyfile
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
import tensorflow as tf
import tf_util
import load_policy

sys.path.append("./../")
from logger import Logger


class policy_network(nn.Module):

    def __init__(self, opt):

        super(policy_network, self).__init__()

        self.input_size = opt.input_size
        self.output_size = opt.output_size

        self.fc1 = nn.Linear(self.input_size, opt.hidden_size)
        self.nonlin1 = nn.PReLU()
        self.fc2 = nn.Linear(opt.hidden_size, self.output_size)

    def forward(self, features):

        x = self.nonlin1(self.fc1(features))
        x = self.fc2(x)

        return x


class Rldataset(data.Dataset):

    def __init__(self, opt, data_file):

        _data = pickle.load(open(data_file, "rb"))

        self.opt = opt
        self.observations = _data['observations']
        self.actions = _data['actions']
        self.len = _data['actions'].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        observations = self.observations[index]
        actions = self.actions[index]

        return observations, actions


def load_data(args, data_file):
    data = Rldataset(args, data_file)
    data_loader = DataLoader(data, batch_size=args.batch_size, num_workers=1, shuffle=True)
    return data_loader


def grad_step(observations, actions, model, optim):
    optim.zero_grad()

    predictions = model(observations)
    loss = torch.mean((predictions - actions)**2)
    loss.backward()
    optim.step()

    return loss


def test(args, model, samples):
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = env.spec.timestep_limit
        policy_fn = load_policy.load_policy(args.expert_policy_file)

        returns = []
        observations = []
        actions = []
        exp_actions = []
        for i in range(samples):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = model(torch.Tensor(obs).unsqueeze(0)).data.numpy()
                exp_action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                exp_actions.append(exp_action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps >= max_steps:
                    break
            returns.append(totalr)

    return returns, observations, exp_actions


def imitation_learning(args):
    logger = Logger('./logs/' + args.log_name)
    data_loader = load_data(args, './data/{}.p'.format(args.envname))
    model = policy_network(args)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    step = 0

    for epoch in range(args.epochs):

        print_loss, tic = 0, time()

        for i, sample in enumerate(data_loader):
            observations, actions = sample
            observations, actions = Variable(observations).float(), Variable(actions).float().squeeze()
            batch_loss = grad_step(observations, actions, model, optimizer)

            print_loss += batch_loss

            if (step+1) % 100 == 0:
                print("Epoch {}, Step {}/{} | Loss: {} | Time: {}".format(epoch+1, step+1, args.epochs*len(data_loader),
                                                                          print_loss / 100, time() - tic))
                logger.scalar_summary('batch_loss', batch_loss, step)
                print_loss, tic = 0, time()

            step = step + 1

    returns = np.asarray(test(args, model, args.test_runs)[0])
    print("Average test reward: {}, std: {}".format(np.average(returns), np.std(returns)))

    return logger, model, optimizer


def combine_data(args, observations, actions):

    _data = pickle.load(open('./data/{}-DAgger.p'.format(args.envname), "rb"))
    _data['observations'] = np.concatenate([_data['observations'], np.array(observations)], axis=0)
    _data['actions'] = np.concatenate([_data['actions'], np.array(actions)], axis=0)
    pickle.dump(_data, open('./data/{}-DAgger.p'.format(args.envname), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def DAgger(args):

    logger, model, optimizer = imitation_learning(args)
    copyfile('./data/{}.p'.format(args.envname), './data/{}-DAgger.p'.format(args.envname))

    for round in range(args.DAgger_rounds):

        returns, observations, exp_actions = test(args, model, args.DAgger_samples)
        combine_data(args, observations, exp_actions)
        data_loader = load_data(args, './data/{}-DAgger.p'.format(args.envname))
        step = 0

        for epoch in range(args.DAgger_epochs):

            print_loss, tic = 0, time()

            for i, sample in enumerate(data_loader):
                observations, actions = sample
                observations, actions = Variable(observations).float(), Variable(actions).float().squeeze()
                batch_loss = grad_step(observations, actions, model, optimizer)

                print_loss += batch_loss

                if (step + 1) % 100 == 0:
                    print("Round {}, Epoch {}, Step {}/{} | Loss: {} | Time: {}".format(round+1, epoch+1, step + 1,
                                                                              args.DAgger_epochs * len(data_loader),
                                                                              print_loss / 100, time() - tic))
                    logger.scalar_summary('batch_loss', batch_loss, step)
                    print_loss, tic = 0, time()

                step = step + 1

        returns = np.asarray(test(args, model, args.test_runs)[0])
        print("Average test reward: {}, std: {}".format(np.average(returns), np.std(returns)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--log_name', type=str, default='test')
    parser.add_argument('--envname', type=str, default='Humanoid-v1')
    parser.add_argument('--expert_policy_file', type=str, default='./experts/Humanoid-v1.pkl')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_runs', type=int, default=10)
    parser.add_argument('--DAgger_rounds', type=int, default=5)
    parser.add_argument('--DAgger_epochs', type=int, default=2)

    args = parser.parse_args()

    if args.envname == "Hopper-v1":
        args.input_size = 11
        args.output_size = 3
        args.hidden_size = 64
        args.lr = 0.005
        args.DAgger_samples = 20
    elif args.envname == "Walker2d-v1":
        args.input_size = 17
        args.output_size = 6
        args.hidden_size = 64
        args.lr = 0.002
        args.DAgger_samples = 20
    elif args.envname == "Humanoid-v1":
        args.input_size = 376
        args.output_size = 17
        args.hidden_size = 256
        args.lr = 0.001
        args.DAgger_samples = 100

    DAgger(args)