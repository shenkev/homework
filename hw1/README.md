# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.

## Result Notes
- it was a little counter intuitive but I had to reduce the number of layers and the hidden size to get BC to start working. On the Hopper-v1 task, it seems like you can overfit quite easily and perform badly at test time.
- Adam worked significantly better than SGD, although you still had to tune the learning rate
- intermediate batch size ~100 is good
- it was obviously much easier to learn Hopper than Humanoid since Humanoid has much larger state and action spaces
- Humanoid is still learnable but you need: more samples, larger neural network, DAgger
- You can learn Humanoid with just BC but it seems to have a large variance in the reward. Either the agent does perfectly or it fails immediately. Training using DAgger stabilizes the agent, most likely because seeing samples from its own policy helps it learn to recover.
