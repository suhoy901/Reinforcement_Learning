import os
import numpy as np
from game import Game
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
import pylab
EPISODES = 10000000

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005

        # 정책신경망과 가치신경망 생성
        self.actor, self.critic = self.build_model()

        self.actor_updator = self.actor_optimizer()
        self.critic_updator = self.critic_optimizer()

        if self.load_model:
            self.actor.load_weights("./save_model/driving_actor_34200.h5")
            self.critic.load_weights("./save_model/driving_critic_34200.h5")

    # actor와 critic의 네트워크 앞의 레이어를 공유하면 조금 더 안정적으로 학습함
    def build_model(self):
        state_input = Input(shape=[self.state_size])
        fc0 = Dense(30, activation='relu', kernel_initializer='he_uniform')(state_input)
        fc1 = Dense(30, activation='relu', kernel_initializer='he_uniform')(fc0)

        policy = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(fc0)
        value = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(fc1)

        actor = Model(inputs=state_input, outputs=policy)
        critic = Model(inputs=state_input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage],[],
                           updates=updates)
        return train

    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])
        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)
        return train

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        self.actor_updator([state, act, advantage])
        self.critic_updator([state, target])

if __name__ == "__main__":

    graph_path = os.path.join(os.getcwd(), 'save_graph')
    model_path = os.path.join(os.getcwd(), 'save_model')

    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    state_size = 5
    action_size = 3
    agent = A2CAgent(state_size, action_size)

    env = Game(6, 10, show_game=False) # True

    # 액터-크리틱(A2C) 에이전트 생성
    scores = []
    episodes = []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while True:

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # policy를 출력해보면 어느정도 수렴했는 지 알 수 있습니다.
            policy = agent.actor.predict(state, batch_size=1).flatten()
            policy = [round(i, 2) for i in policy]

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/driving_a2c.png")
                print("episode:", e, "  score:", score)

                if e % 100 == 0:
                    mean_score = np.mean(scores[-min(len(scores), 100):])
                    print("episode:", e, " mean score :", mean_score, "logit : ", policy)
                    agent.actor.save_weights("./save_model/driving_actor.h5")
                    agent.critic.save_weights("./save_model/driving_critic.h5")
                break