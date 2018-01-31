import time
import gym
import gym_maze

# 환경 선언 및 초기화
env = gym.make('maze-sample-10x10-v0')
env.reset()

# action_space
print(env.action_space)
print(env.action_space.n)
print(type(env.action_space))

# 연속 행동
for i in range(100):

    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    print(action, state, reward, done)
    # state 는 [x축 방향 오른 쪽으로 몇칸, y축 방향 아래로 몇칸]
    # action (0 - 위) (1 - 아래) (2 - 오른쪽) (3 - 왼쪽)
    env.render()
