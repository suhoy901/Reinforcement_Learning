import copy
import pylab
import random
import numpy as np
from environment_10by10 import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import os

EPISODES = 1000


class LinearAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # 상태의 크기와 행동의 크기 정의
        self.action_size = action_size
        self.state_size = state_size
        self.discount_factor = 0.99
        self.learning_rate = 0.01

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.1
        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.01
            self.model.load_weights('./save_model/linear_10by10.h5')

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.action_size,
                        input_dim=self.state_size,
                        activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return random.randrange(self.action_size)
        else:
            # 모델로부터 행동 산출
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    # SARSA
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]

        # 살사의 큐함수 업데이트 식
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        # 출력 값 reshape
        target = np.reshape(target, [1, self.action_size])
        # 인공신경망 업데이트
        self.model.fit(state, target, epochs=1, verbose=0)


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    state_size = env.state_size[0]
    action_size = env.action_size

    agent = LinearAgent(state_size, action_size)

    if not os.path.isdir('./save_model'):
        os.mkdir('./save_model')
    if not os.path.isdir('./save_graph'):
        os.mkdir('./save_graph')

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        score = 0

        state = env.reset(render=agent.render)
        state = np.reshape(state, [1, state_size])

        while True:
            # env 초기화
            if agent.render:
                env.render()

            global_step += 1

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)
            # 샘플로 모델 학습
            agent.train_model(state, action, reward, next_state, next_action,
                              done)

            state = next_state
            score += reward

            if done:
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)

                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

                if e % 50 == 0:
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./save_graph/linear_10by10.png")

                if np.mean(scores[-min(10, len(scores)):]) > 99 \
                        or agent.epsilon < agent.epsilon_min:
                    agent.model.save_weights("./save_model/linear_10by10.h5")
                    agent.render = True
                else:
                    agent.render = False

                break
