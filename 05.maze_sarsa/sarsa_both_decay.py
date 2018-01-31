import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.pyplot import ylim
import os
import numpy as np
import random
from collections import defaultdict
import gym
import gym_maze
np.random.seed(1)

plt.style.use('ggplot')
ylim((-2, 1))
env = gym.make('maze-sample-10x10-v0')

# State 의 boundary
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# Maze의 size (10, 10)
NUM_GRID = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
# action space
ACTION = ['up', 'dw', 'ri', 'le']

# gui환경의 Render 여부
RENDER = False


class SarsaAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.9
        self.discount_factor = 0.9
        self.epsilon = 1.
        self.e_step = (1. - 0.01) / 100
        self.lr_step = (0.9 - 0.2) / 100
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s', a'>의 샘플로부터 큐함수를 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate *
                 (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q

    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state_action = self.q_table[str(state)]
            action = self.arg_max(state_action)
        return int(action)

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # greedy 정책 출력
    def print_policy(self):
        for y in range(NUM_GRID[0]):
            for x in range(NUM_GRID[1]):
                print("%s" % ACTION[self.arg_max(self.q_table[str((x, y))])], end=" ")
            print("")


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_GRID[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_GRID[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_GRID[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":
    # 환경 선언 및 초기화
    env.reset()
    agent = SarsaAgent(actions=list(range(env.action_space.n)))
    scores = []
    episodes = []

    for episode in range(250):
        state = env.reset()
        state = state_to_bucket(state)
        action = agent.get_action(state)
        total_reward = 0

        while True:
            if RENDER:
                env.render()

            # <s,a,r,s',a'> sample 수집
            next_state, reward, done, _ = env.step(action)
            next_state = state_to_bucket(next_state)
            next_action = agent.get_action(next_state)

            # <s,a,r,s',a'>로 큐함수를 업데이트
            agent.learn(str(state), action, reward, str(next_state), next_action)
            total_reward += reward
            state = next_state
            action = next_action

            if done:
                print("Episode : %d total reward = %f . " % (episode, total_reward))
                print(agent.learning_rate, agent.epsilon)
                agent.epsilon -= agent.e_step
                agent.learning_rate -= agent.lr_step
                episodes.append(episode)
                scores.append(total_reward)

                if agent.learning_rate < 0.2:
                    agent.learning_rate = 0.2

                if agent.epsilon < 0.01:
                    agent.epsilon = 0.01

                if episode % 50 == 0:
                    if not os.path.isdir('./save_graph'):
                        os.mkdir('./save_graph')
                    plt.plot(episodes, scores)
                    plt.savefig("./save_graph/sarsa_both_decay.png")
                break

        if np.mean(scores[-min(10, len(scores)):]) > 0.93:
            RENDER = True
            agent.print_policy()
        else:
            RENDER = False
