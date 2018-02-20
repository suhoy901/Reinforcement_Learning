import os
import sys
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense, Input, Lambda, Add
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K

plt.style.use('ggplot')
np.random.seed(0)
EPISODES = 300
global_steps = 0

model_path = os.path.join(os.getcwd(), 'save_model')
if not os.path.isdir(model_path):
    os.mkdir(model_path)

graph_path = os.path.join(os.getcwd(), 'save_graph')
if not os.path.isdir(graph_path):
    os.mkdir(graph_path)


# 카트폴 예제에서의 Dualing DDQN 에이전트
class DualingDDQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.999
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dueling_ddqn_trained.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):

        input_x = Input(shape=(self.state_size,))
        shared = Dense(24, input_dim=self.state_size, activation='relu',
                       kernel_initializer='he_uniform')(input_x)

        advantage_fc = Dense(24, input_dim=self.state_size, activation='relu',
                             kernel_initializer='he_uniform')(shared)

        value_fc = Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform')(shared)

        advantage = Dense(self.action_size, activation='linear',
                          kernel_initializer='he_uniform')(advantage_fc)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                           output_shape=(self.action_size,))(advantage)

        value = Dense(1, activation='relu',kernel_initializer='he_uniform')(value_fc)

        q_value = Add()([value, advantage])
        model = Model(input=input_x, output=q_value)

        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), np.max(self.model.predict(state))
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0]), np.max(q_value)

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size))
        update_target = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # DDQN의 핵심 파트
                # 행동의 선택은 현재 모델이 고른 대로 선택
                # 큐 함수 자체는 target 으로 부터 가져옴.
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


if __name__ == "__main__":
    graph_path = os.path.join(os.getcwd(), 'save_graph')
    model_path = os.path.join(os.getcwd(), 'save_model')

    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN 에이전트 생성
    agent = DualingDDQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while True:
            if agent.render:
                env.render()

            # 현재 상태로 행동을 선택
            action, _ = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or score == 499 else -100

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state
            global_steps += 1

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                score = score if score == 500 else score + 100
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                plt.plot(episodes, scores)
                plt.savefig("./save_graph/cartpole_dueling_ddqn_dump.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon, "global steps: ", global_steps)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.model.save_weights("./save_model/cartpole_dueling_ddqn_dump.h5")
                    agent.render = True
                    sys.exit()

                else:
                    agent.render = False

                break
