from PIL import Image
import gym
import gym_pacman
import time

env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)


done = False

while True:
    done = False
    env.reset("mediumClassic")
    i = 0
    while i < 100:
        i += 1
        s, r, done, info = env.step(env.action_space.sample())
        print(s.shape)
        env.render()
        
