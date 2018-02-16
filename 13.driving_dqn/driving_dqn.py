import os
import matplotlib
matplotlib.use('TkAgg')
import pylab
import random
import numpy as np
from collections import deque
from game import Game
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        # 상태와 행동의 크기를 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN하이퍼파리미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.1
        self.batch_size = 50
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기를 2000으로 설정함
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 설정
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃모델 초기화
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/driving_t.h5")
            self.epsilon = 0.01

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 엡실론 탐욕 정책으로 행동을 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델을 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == "__main__":
    graph_path = os.path.join(os.getcwd(), 'save_graph')
    model_path = os.path.join(os.getcwd(), 'save_model')

    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    state_size = 5
    action_size = 3

    # DQN에이전트 생성
    agent = DQNAgent(state_size, action_size)

    env = Game(6, 10, show_game=agent.load_model)

    scores, episodes = [], []

    for e in range(EPISODES):
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while True:
            # 현재 상태로 행동을 선택함
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # 리플라이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)

            # 매 타입스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이터
                agent.update_target_model()
                # 에피소드마다 학습 결과를 출력
                scores.append(score)
                episodes.append(e)

                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/driving_a2c.png")
                print("episode:",e,"    score:", score, "   memory length:"
                      ,len(agent.memory), " epsilon:", agent.epsilon)

                if agent.epsilon < agent.epsilon_min:
                    agent.model.save_weights("./save_model/driving.h5")
                break


