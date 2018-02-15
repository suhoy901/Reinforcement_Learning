import sys
import os
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 1000

class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # 화면을 실행시킬 지 여부와 저장된 모델을 불러올지 여부를 설정
        self.render = False
        self.load_model = False

        # state_size와 action_size를 설정해준다
        self.state_size = state_size
        self.action_size = action_size

        # 학습을 위한 하이퍼파라미터를 설정
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()

        # 에피소드 단위로 학습하기 위해 states와 actios와 rewards를 클래스 변수로 선언
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_reinforce.h5")


    # policy를 근사시키는 네트워크임
    # state를 input을 받고 각 행동에 대한 확률인 policy를 출력하는 모델임
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='glorot_uniform')) # Xavier초기값
        model.add(Dense(24, activation="relu", kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation="softmax", kernel_initializer='glorot_uniform'))
        model.summary()

        # Categorical crossentropy를 사용하는 이유는 policy gradient의 loss를
        # 쉽게 계산하기 위함임. categorical_crossentropysms sum(p_i * log(q_i))으로
        # 정의됨. 여기서 p_i는 Gt q_i는 실제 output으로 출력된 policy에서 현재 행동에
        # 해당하는 확률임
        model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=self.learning_rate))
        return model

    # 모델의 아웃풋에서 행동을 확률에 따라 선택함
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # Gt에 해당하는 Return을 구하는 함수임
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 한 에피소드 동안 s, a, r을 저장함
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)


    # 매 에피소드 마다 모델을 훈련
    def train_model(self):
        episode_length = len(self.states)

        # Gt를 노멀라이즈 함
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        # input과 Gt를 numpy zeros로 초기화함
        update_inputs = np.zeros((episode_length, self.state_size))
        Gt = np.zeros((episode_length, self.action_size))

        # 현재 action에 해당하는 index에 Gt를 넣어주어 곱했을 때 나머지가 0이 되도록 함
        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            Gt[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, Gt, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [],[],[]

if __name__ == "__main__":

    graph_path = os.path.join(os.getcwd(), 'save_graph')
    model_path = os.path.join(os.getcwd(), 'save_model')

    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    env = gym.make('CartPole-v1')
    # 환경으로부터 state_size와 action_size를 받아온다
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # REINFORCE에이전트를 선언
    agent = REINFORCEAgent(state_size, action_size)
    scores = []
    episodes = []

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

            # 500 time-step을 끝내지 못하고 끝났을 시 -100의 보상을 설정
            reward = reward if not done or score == 499 else -100

            # 한 에피소드 동안의 <s, a, r> 샘플을 저장
            agent.append_sample(state, action, reward)

            score += reward
            state = next_state

            if done:
                # 매 에피소드마다 모은 sample로 agent를 학습한다
                agent.train_model()

                # 모든 에피소드마다 plot
                # 500을 채우지 못하고 끝난 경우에 대해 -100을 해준 것을 빼고 점수를 저장
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole-reinforce.png")
                print("episode:", e, "  score:", score)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()
                break

        # save the model
        if e % 50 == 0:
            agent.model.save_weights("./save_model/cartpole_reinforce.h5")


