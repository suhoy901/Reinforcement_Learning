import time
import gym
import gym_maze

env = gym.make("maze-sample-10x10-v0")

env.reset()

for i in range(100):
    env.step(env.action_space.sample())
    time.sleep(0.1)
    env.render()