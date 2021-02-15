import sys
import torch
import json
import gym
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from train import train_agent
from framestack import FrameStack
import gym_pacman
from utils import time_format, eval_policy, mkdir
from agent import Agent

def main(args):
    """ """
    print(sys.version)
    with open (args.param, "r") as f:
        param = json.load(f)

    print("use the env {} ".format(param["env_name"]))
    print(param)
    print("Start Programm in {}  mode".format(args.mode))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    param["locexp"] = args.locexp
    env = gym.make(param["env_name"])
    env  = FrameStack(env, param)
    agent = Agent(args, env)
    agent.load(str(args.locexp), "1/checkpoint-52038.pth")
    pathname = args.model_path
    tensorboard_name = 'eval_agents' + '/runs/' + pathname
    writer = SummaryWriter(tensorboard_name) 
    T = 100  
    eval_policy(env, agent, memory, T, param)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="hypersearch", type=str)
    parser.add_argument('--id', type=str, default="1", help='Experiment ID')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(50), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=3, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.9, metavar='σ', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=1, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--learning-rate', type=float, default=5e-4, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    parser.add_argument('--learn-start', type=int, default=int(1e3), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
    parser.add_argument('--checkpoint-interval', default=10000, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--mode', default='train', type= str, help='choose mode')
    parser.add_argument('--model_path', default='', type= str, help='path for trained weights')
    arg = parser.parse_args()
    main(arg)
