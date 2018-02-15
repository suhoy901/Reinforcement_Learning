import os
import sys
import gym
import pylab
import numpy as np

from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000


# global_steps = 0

class A2CAgent:
    def __init__(self, state_size, action_size):
        # 화면을 실행시킬지 여부와 저장된 모델을 불러올 지 여부를 설정함
        self.render = False
        self.load_model = False

        # state_size와 action_size 그리고 critic에서 출력될 value의 size를 설정
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # 학습을 위한 hypter parameter를 설정
        # actor가 더 빠르게 수렴하는 경향이 있기 때문에 lr을 더 작게 잡아줌
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_actor.h5")
            self.critic.load_weights("./save_model/cartpole.critic.h5")

    # policy를 근사키시는 actor네트워크
    # actor는 state를 input으로 받고 각 행동에 대한 확률인 policy를 출력으로 하는 모델
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # Critic은 상태를 input으로 받아 상태에 대한 value를 출력하는 모델
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(24, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))
        return critic

    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]


    def train_model(self, state, action, reward, next_state, done):
        # target과 advantage를 numpy_zero로 선언
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        # critic모델로부터 현재 state에 대한 value와 next_state에 대한 value를 산출
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            # [[a0, a1, a2]]
            advantages[0][action] = reward - value
            # [[value]]
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)


if __name__=="__main__":
    graph_path = os.path.join(os.getcwd(), 'save_graph')
    model_path = os.path.join(os.getcwd(), 'save_model')

    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    env = gym.make('CartPole-v1')
    # 환경으로부터 state_size와 action_size를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # A2C 에이전트를 선언
    agent = A2CAgent(state_size, action_size)
    scores, episodes = [], []


    for e in range(EPISODES):
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while True:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # 500 time-step을 끝내지 못하고 끝났을 때 -100의 보상을 설정한다
            reward = reward if not done or score == 499 else -100

            # 모델을 step by step으로 학습
            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # 모든 에피소드마다 plot함
                # 500을 채우지 못하고 끝난 경우에 대해 -100을 해준 것을 빼고 점수를 저장
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig('./save_graph/cartpole_a2c.png')
                print("episode:", e, "   score:", score)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()
                break

        # save the model
        if e % 50 == 0:
            agent.actor.save_weights("./save_model/cartpole_actor.h5")
            agent.critic.save_weights("./save_model/cartpole_critic.h5")
