from environment import Env
import numpy as np
import time
from collections import defaultdict

# env 선언 및 reset
env = Env()
env.reset()

# action space 출력
print(env.n_actions)

# 에이전트의 행동을 환경에 전달하고 정보 받아오기
next_state, reward, done = env.step(1)
print(next_state, reward, done)

# q 함수를 담을 default dict
q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
q_table[str([1, 0])] = [100., 100., 100., 100.]
print(q_table['blah blah'])
print(q_table)

# 환경에 q함수를 출력
env.print_value_all(q_table)
# 환경 업데이트
env.render()
time.sleep(3)

# 에이전트 연속 행동
for i in range(100):
    actions = np.random.choice(4)
    env.step(actions)
    env.render()
env.reset()

env.destroy()
