import gym
import random
import numpy as np

# envids = [spec.id for spec in gym.envs.registry.all()]
# print(len(envids))
# for envid in sorted(envids):
#     print(envid)


print("\n\n")
# game = random.choice(envids)
game = 'BreakoutDeterministic-v4'
env = gym.make(game)

print("환경 : ", env)
print("행동 : ",env.action_space)
print("상태 : ", env.observation_space)
print("보상 : ",env.reward_range)
print("행동의 갯수 : ", env.action_space.n)


for i_episode in range(1):
    observation = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        print("observation :", observation.shape)
        print("reward :", reward)
        print("done : ", done)
        print("info : ", info)
        if done:
            print("Episode finish")
            break

print("finished")

print("\n\n액션 테스트")
for e in range(env.action_space.n):
    observation = env.reset()
    print("test for action : ", e)

    for _ in range(500):
        env.render()
        action = e
        env.step(action)

env.close()