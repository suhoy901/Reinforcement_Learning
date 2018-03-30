# Reinforcement Learning
- 학습전 이해 : MP, DP, SP, MDP(https://norman3.github.io/rl/)
- n-step, episode, sequence
- deterministic policy vs stochastic policy
- policy 2가지 형태
  - 정책(상태) -> 명시적(explicit) -> Policy Iteration
  - 선택(가치함수(상태)) -> 내재적(implicit) -> Value Iteration
----

## 내용
### A. DynamicProgramming
- value_iteration
  - [gridworld예제 : value_iteration](https://github.com/suhoy901/Reinforcement_Learning/blob/master/01.gridworld_value_iteration/value_iteration.py)
- monte-carlo
  - [gridworld예제 : Monte-Carlo](https://github.com/suhoy901/Reinforcement_Learning/blob/master/02.monte-carlo/environment.py)

### B. 고전적 강화학습
- sarsa
  - [gridworld예제 : SARSA](https://github.com/suhoy901/Reinforcement_Learning/blob/master/03.gridworld_sarsa/sarsa_agent.py)
  - [maze예제 : SARSA](https://github.com/suhoy901/Reinforcement_Learning/blob/master/05.maze_sarsa/sarsa_basic.py)
  - [maze예제 : SARSA_epsilon_decay](https://github.com/suhoy901/Reinforcement_Learning/tree/master/05.maze_sarsa)
  - [maze예제 : SARSA_lr_decay](https://github.com/suhoy901/Reinforcement_Learning/blob/master/05.maze_sarsa/sarsa_lr_decay.py)
  - [maze예제 : SARSA_both_decay](https://github.com/suhoy901/Reinforcement_Learning/blob/master/05.maze_sarsa/sarsa_both_decay.py)

- q-learning :
  - [gridworld예제 : Q-Learning](https://github.com/suhoy901/Reinforcement_Learning/blob/master/04.gridworld_q_learning/q_learning_agent.py)
  - [maze예제 : Q-Learning](https://github.com/suhoy901/Reinforcement_Learning/blob/master/06.maze_q_learning/q_learning_basic.py)
  - [maze예제 : Q-Learning_both_decay](https://github.com/suhoy901/Reinforcement_Learning/blob/master/06.maze_q_learning/q_learning_both_decay.py)

### C. 수정



## Keyword
### 1. Markov Decision Process
- MDP : State, Action, State transition probability, Reward, Discount_Factor
- 단기적 보상 vs 장기적 보상 : Sparse reward, delayed reward문제를 극복 -> 반환값(Return), 가치함수(Value function)
  - Return : 현재가치(PV)로 변환한 reward들의 합
  - Value function(가치함수) : Return에 대한 기대값
    - State Value Function(상태가치함수) : Q-function기대값이며 policy사용, 반환값(보상,감가율), 상태
    - Q-function(행동가치함수) : 특정 state에서 a라는 action을 취할 경우의 받을 return에 대한 기대값 

### 2. Bellman Equation
- Bellman Expectation Equation(벨만기대방정식) : 특정 정책에 대한 가치함수<br>
  - 가치함수에 대한 벨만기대방정식 : ![벨만기대방정식](http://latex.codecogs.com/gif.latex?%5Cnu_%7B%5Cpi%20%7D%28s%2C%20a%29%20%3D%20E_%7B%5Cpi%7D%5BR_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20%5Cnu_%7B%5Cpi%7D%28S_%7Bt&plus;1%7D%29%20%7C%20S_t%20%3D%20s%5D)
  - 큐함수에 대한 벨만기대방정식 : ![벨만기대방정식](http://latex.codecogs.com/gif.latex?q_%7B%5Cpi%20%7D%28s%2C%20a%29%20%3D%20E_%7B%5Cpi%7D%5BR_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20q_%7B%5Cpi%7D%28S_%7Bt&plus;1%7D%2C%20A_%7Bt&plus;1%7D%29%20%7C%20S_t%20%3D%20s%2C%20A_t%20%3D%20a%5D)
- Bellman Optimality Equation : optimal value function사이의 관계식
  - Optimal value function
  - 가치함수에 대한 벨만최적방정식 : ![벨만최적방정식](http://latex.codecogs.com/gif.latex?%5Cnu%5E*_%7B%5Cpi%20%7D%28s%29%20%3D%20max_a%20E_%7B%5Cpi%7D%5BR_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20%5Cnu%5E%7B*%7D%28S_%7Bt&plus;1%7D%29%20%7C%20S_t%20%3D%20s%2C%20A_t%20%3D%20a%5D)
  - 큐함수에 대한 벨만최적방정식 : ![벨만최적방정식](http://latex.codecogs.com/gif.latex?q%5E*_%7B%5Cpi%20%7D%28s%2C%20a%29%20%3D%20E_%7B%5Cpi%7D%5BR_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20max_%7Ba%27%7D%20q%5E*_%7B%5Cpi%7D%28S_%7Bt&plus;1%7D%2C%20a%27%29%20%7C%20S_t%20%3D%20s%2C%20A_t%20%3D%20a%5D)
  
### 3. Dynamic Programming : Model-Base
- 큰 문제를 작은 문제로, 반복되는 값을 저장하면서 해결
  - 큰 문제 : 최적가치함수 계산 ![](http://latex.codecogs.com/gif.latex?%5Cnu_0%20%5Crightarrow%20%5Cnu_1%20%5Crightarrow%20%5Cnu_2%20%5Crightarrow%20%5Cnu_3%20%5Crightarrow%20...%20%5Crightarrow%20%5Cnu%5E*)
  - 작은 문제 : 현재의 가치함수를 더 좋은 가치함수로 업데이트 ![](http://latex.codecogs.com/gif.latex?%5Cnu_k%20%5Crightarrow%20%5Cnu_%7Bk&plus;1%7D)
  - 벨만방적식으로 1-step계산으로 optimal계산
- Value Iteration(Bellman Optimality Equation)
  - 가치함수가 최적이라는 과정 : ![](http://latex.codecogs.com/gif.latex?%5Cnu_%7Bk&plus;1%7D%28s%29%20%5Cleftarrow%20max_a%5BR_s%5Ea%20&plus;%20%5Cgamma%20%5Cnu_k%28s%27%29%5D)
  - 수렴한 가치함수에 greedy policy
  - Q-Learning

- Policy Iteration(Bellman Expectation Equation) : 정책평가 + 정책발전, GPI
  - 벨만 기대방정식을 이용함. ![](http://latex.codecogs.com/gif.latex?%5Cnu_%7Bk&plus;1%7D%28s%29%20%5Cleftarrow%20%5CSigma_%7Ba%5Cin%20A%7D%20%5Cpi%28a%7Cs%29%5BR_s%5Ea%20&plus;%20%5Cgamma%20%5Cnu_k%28s%27%29%5D)
  - evaluation(Prediction) : 벨만기대방성식을 이용한 참 가치함수를 계산, 정책 파이에 대한 참 가치함수를 반복적으로, 모든 상태에 대해 동시에(한번)
  - improvement(Control) : 가치함수로 정책 파이를 업데이트, greedy policy
  - SARSA
 
### 4. Reinforcement(고전적) : Model-Free(sampling)
- Off-policy vs On-policy
- **SARSA** : s,a,r,s',a' : Policy Iteration에 sampling적용
  - policy Iteration vs SARSA
    - 정책평가 -> TD-Learning
    - 정책발전 -> 엡실론 탐욕
  - 정책평가 : TD Learning(Bootstrap)
    1. 에이전트가 현재 상태 S_t=s에서 행동 A_t=a 선택하고 S,A
    2. 선택한 행동으로 환경에서 1-step진행
    3. 다음 상태 S_(t+1)=s'와 보상 R_(t+1)을 환경으로부터 받음 R
    4. 다음 상태 S_(t+1)=s'에서 행동 S', A'
    - ![](http://latex.codecogs.com/gif.latex?q%28s%2C%20a%29%20%3D%20q%28s%2C%20a%29%20&plus;%20%5Calpha%28r%20&plus;%20%5Cgamma%20q%28s%27%2C%20a%27%29%20-%20q%28s%2Ca%29%29) 
    
  - 정책발전 improvement(Control) => 엡실론 탐욕 : 엡실론의 확률로 최적이 아닌 행동을 선택
    - 엡실론 탐욕정책으로 샘플을 획득함
  - 문제점 : on-policy(안좋은 보상을 만날경우 큐함수 업데이트가 지속적 감소) 
  
  
- **Q-Learning** : Value Iteration에 sampling적용, Off-Policy(2개의 정책) -> s,a,r,s'
  - tip) off-policy : behavior policy(샘플수집정책 : 업데이트X), target policy(에이전트의 정책:업데이트o)
  - 벨만최적방정식으로 큐함수 업데이트, off-policy
    - ![](http://latex.codecogs.com/gif.latex?q%28s%2C%20a%29%20%3D%20q%28s%2C%20a%29%20&plus;%20%5Calpha%28r%20&plus;%20%5Cgamma%20%7B%5Ccolor%7BRed%7Dmax%20%7D_%7Ba%27%7D%20q%28s%27%2C%20a%27%29%20-%20q%28s%2Ca%29%29)
  - Off-policy Learning
    - 행동하는 정책(exploration) : 엡실론탐욕, 볼츠만 등..
    - 학습하는 정책(exploitation) : 탐욕정책
  - 요약 : Q-Learning을 통한 학습 과정
    1. 상태 s에서 행동 a는 행동정책(엡실론 탐욕)으로 선택
    2. 환경으로부터 다음 상태 s'와 보상 r을 받음
    3. 벨만 최적방정식을 통해 q(s,a)를 업데이트
      - 학습정책 : 탐욕정책
      - ![](http://latex.codecogs.com/gif.latex?q%28s%2C%20a%29%20%3D%20q%28s%2C%20a%29%20&plus;%20%5Calpha%28r%20&plus;%20%5Cgamma%20%7B%5Ccolor%7BRed%7Dmax%20%7D_%7Ba%27%7D%20q%28s%27%2C%20a%27%29%20-%20q%28s%2Ca%29%29)
- SARSA vs Q-Learning
  - on-policy TD Learning vs off-policy TD Learning
  - Update target :
    - ![](http://latex.codecogs.com/gif.latex?r%20&plus;%20%5Cgamma%20q%28s%27%2C%20a%27%29) vs ![](http://latex.codecogs.com/gif.latex?r%20&plus;%20%5Cgamma%20%7B%5Ccolor%7BRed%7Dmax%20%7D_%7Ba%27%7D%20q%28s%27%2C%20a%27%29)
    
    
    
### 5. Value Function Approximation
- Value function Approximation
  - Large state space, 비슷한 state는 비슷한 function의 output -> Generalization
  - Supervised Learning기법을 사용하여 Generalization
    - Function Approximation : Target function을 approximate하는 function을 찾음
      - Target function을 아는 경우 : 수치해석학(numerical analysis)
      - Target function을 모르는 경우 : regression, classification, ...
    - MSE
      - 큐함수의 결과값은 continous(regression), MSE, loss function(**TD-error**)
      - MSE : ![](http://latex.codecogs.com/gif.latex?%28r%20&plus;%20%5Cgamma%20q_%7B%5Ctheta%7D%28s%27%2C%20a%27%29%20-%20q_%7B%5Ctheta%7D%28s%2C%20a%29%29%5E2)
    - Gradient Descent
      - MSE의 Gradient : ![](http://latex.codecogs.com/gif.latex?-2%28%7B%5Ccolor%7BRed%7D%20target%7D%20-%20q_%7B%5Ctheta%7D%28s%2C%20a%29%29%20%5Cnabla_%7B%5Ctheta%7D%20q_%7B%5Ctheta%7D%28s%2Ca%29)
      - lr, target적용 : ![](http://latex.codecogs.com/gif.latex?-%5Calpha%28%7B%5Ccolor%7BRed%7D%20r%20&plus;%20%5Cgamma%20q_%7B%5Ctheta%7D%28s%27%2C%20a%27%29%7D%20-%20q_%7B%5Ctheta%7D%28s%2C%20a%29%29%20%5Cnabla_%7B%5Ctheta%7D%20q_%7B%5Ctheta%7D%28s%2Ca%29)
    - 새로운 파라미터 : 기존parameter - (lr)(MSE의 graduent)
  - SARSA with function appoximation
    - ![](http://latex.codecogs.com/gif.latex?%5Ctheta%20%5Cleftarrow%20%5Ctheta%20-%5Calpha%28r%20&plus;%20%5Cgamma%20q_%7B%5Ctheta%7D%28s%27%2C%20a%27%29%20-%20q_%7B%5Ctheta%7D%28s%2C%20a%29%29%20%5Cnabla_%7B%5Ctheta%7D%20q_%7B%5Ctheta%7D%28s%2Ca%29)
  - Q-Learning with function approximation
    - ![](http://latex.codecogs.com/gif.latex?%5Ctheta%20%5Cleftarrow%20%5Ctheta%20-%5Calpha%28r%20&plus;%20%5Cgamma%20max_%7Ba%27%7D%20q_%7B%5Ctheta%7D%28s%27%2C%20a%27%29%20-%20q_%7B%5Ctheta%7D%28s%2C%20a%29%29%20%5Cnabla_%7B%5Ctheta%7D%20q_%7B%5Ctheta%7D%28s%2Ca%29)
  - function approximation 종류(Linear, Nonlinear)
    - linear
      - 큐함수의 gradient
        - ![](http://latex.codecogs.com/gif.latex?%5Cnabla_%7Bw%7D%20q_%7Bw%7D%28s%2Ca%29%20%3D%20%5Cnabla_%7Bw%7D%28X%28s%29W%29%20%3D%20X%28s%29)
      - Q-Learning with function approximation
        - ![](http://latex.codecogs.com/gif.latex?w%20%5Cleftarrow%20w%20-%5Calpha%28r%20&plus;%20%5Cgamma%20max_%7Ba%27%7D%20q_%7Bw%7D%28s%27%2C%20a%27%29%20-%20q_%7Bw%7D%28s%2C%20a%29%29%20%5Cnabla_%7B%5Ctheta%7D%20X%28s%29)
    - nonlinear(Neural net)
      - MSE error에 대한 gradient : ![](http://latex.codecogs.com/gif.latex?%5Cnabla_%7B%5Ctheta%7D%28r%20&plus;%20%5Cgamma%20max_%7Ba%27%7D%20q_%7B%5Ctheta%7D%20%28s%27%2Ca%27%29%20-%20q_%7B%5Ctheta%7D%28s%2Ca%29%29%5E2)
      - 문제점 : sample by sample로 업데이트하기 때문에 다른 상태들의 큐함수에도 영향을 미침, 학습이 잘 안됨
  - Online update vs offline update
    - Online : 에이전트가 환경과 상호작용하고 있는 도중에 update
    - Offline : 에피소드가 끝난 후에 update

## 보강내용
### Value-based RL vs Policy-based RL
- Value-based RL
  - 가치함수를 통해 행동을 선택
  - 가치함수에 function approximator적용
  - 업데이트 대상이 가치함수(큐함수)
  - 각 행동이 얼마나 좋은지 큐함수를 통해 측정
  - 큐함수를 보고 행동을 선택(엡실론그리디)
  - SARSA, Q-LEARNING, DQN,...
  
- Policy-based RL
  - 정책에 따라 행동을 선택
  - 정책에 function approximator적용
  - 업데이트 대상이 정책
  - approximate된 정책의 input은 상태 벡터, output은 각 행동을 할 확률
  - 확률적으로 행동을 선택(stochastic policy)
  - REINFORCE, Actor-Critic, A3C,...


### 6. DQN(Deep Q-Network)
**BreakOut_v4**
**Cartpole_DQN2015**
- DQN2015 특징
  - CNN
  - Experience reply
    - Sample들의 상관관계를 깬다
  - Online update with Stochastic gradient descent
    - 점진적으로 변하는 큐함수에 대한 입실론그리디폴리시 사용
    - epsilon-greedy policy로 탐험하며 reply memory에서 추출한 mini-batch로 큐함수 업데이트
    - Q-learning update
      - ![](http://latex.codecogs.com/gif.latex?q%28s%2C%20a%29%20%3D%20q%28s%2C%20a%29%20&plus;%20%5Calpha%28r%20&plus;%20%5Cgamma%20max_a%20q%28s%27%2Ca%27%29%20-%20q%28s%2C%20a%29%29)
    - DQN update : MSE error를 backpropagation
      - ![](http://latex.codecogs.com/gif.latex?MSE%20error%20%3A%20%28%7B%5Ccolor%7BRed%7D%20r%20&plus;%20%5Cgamma%20max_%7Ba%27%7D%20q_%7B%5Ctheta%5E-%7D%28s%27%2C%20a%27%29%7D%20-%20q%28s%2C%20a%29%29)
  - Target Q-network
     - Target network ![](http://latex.codecogs.com/gif.latex?%5Ctheta%5E-)의 사용 : update의 target이 계속 변하는 문제를 개선
     - 일정주기마다 현재의 network ![](http://latex.codecogs.com/gif.latex?%5Ctheta%5E-)를 ![](http://latex.codecogs.com/gif.latex?%5Ctheta)로 업데이트
       
      
- DQN 학습과정
  - 탐험
    - 정책은 큐함수에 대한 epsilon-greedy policy
    - 엡실론은 time-step에 따라서 decay함
    - 엡실론은 1부터 시작해서 0.1까지 decay함. 이후 0.1을 지속적으로 유지함
  - 샘플의 저장
    - 에이전트는 epsilon-greedy policy에 따라 샘플 s,a,r,s'를 생성함
    - 샘플을 reply memory에 append함
  - 무작위 샘플링
    - 미니배치(32개) 샘플 추출
    - 샘플로부터 target값과 prediction값을 구함(32개)
      - MSE-error : ![](http://latex.codecogs.com/gif.latex?%28target%20-%20prediction%29%5E2)
      - Target : ![](http://latex.codecogs.com/gif.latex?r%20&plus;%20%5Cgamma%20max_%7Ba%27%7D%20q_%7B%5Ctheta%7D%28s%27%2C%20a%27%29)
      - Prediction : ![](http://latex.codecogs.com/gif.latex?q_%7B%5Ctheta%7D%28s%2C%20a%29)
  - 일정 주기마다 Target network 업데이트

- DQN알고리즘 세부내용
  - Image preprocessing
    - Gray-Scale(210,160,1) -> (210,160,1)
    - Resize(210, 150,1) -> (84,84,1)
  - 4 images 1 history
    - 이미지한장에는 속도 정보 포함X
    - 연속된 4개의 image를 하나의 history로 네트워크에 input
    - 학습에 4번의 image중에 1개만 사용함(frame skip)
    - Frame skip한 상황에서 image를 하나의 history로
  - 30 no-op
    - 항상 같은 상태에서 시작하므로 초반에 local optimum으로 수렴할 확률이 높음
    - 0에서 30의 time-step 중에 랜덤으로 하나를 선택한 후 아무것도 안함
  - Reward clip
    - 게임마다 다른 reward를 통일
  - Huber loss
    - -1과 1사이는 quadratic, 다른 곳은 linear

- DQN
 1. 환경초기화, 30 no-op
 2. History에 따라 행동을 선택(엡실론그리디), 엡실론의 값을 decay
 3. 선택한 행동으로 1 time-step환경에서 진행, 다음상태, 보상을 받음
 4. 샘플을 형성(h,a,r,h'), reply memory에 append
 5. 50000스텝 이상일 경우 reply memory에서 mini-batch
  - ![](http://latex.codecogs.com/gif.latex?MSE%20error%20%3A%20%28r%20&plus;%20%5Cgamma%20max_%7Ba%27%7D%20q_%7B%5Ctheta%7D%20-%20%28s%27%2C%20a%27%29%20-%20q_%7B%5Ctheta%7D%28s%2C%20a%29%5E2%20%29)
 6. 10000스텝마다 target network업데이트
 
### 7. Faster DQN
차후에 내용 보강


### 8. REINFORCE
- Policy- based RL학습방법
  - 정책에 따라 행동
  - 행동한 결과에 따라 정책을 평가
    - 기준 : **목적함수(Objectvive function/performance measure)** 
      - ![](https://latex.codecogs.com/gif.latex?J%28%5Ctheta%29%20%3D%20%5Cnu_%7B%5Cpi_%7B%5Ctheta%7D%7D%28s_0%29%20%3D%20E_%7B%5Cpi_%7B%5Ctheta%7D%7D%5B%5CSigma_%7Bt%3D0%7D%5E%7BT-1%7D%20%5Cgamma%20r_%7Bt&plus;1%7D%20%7C%20S_0%20%3D%20s_0%20%5D)
    - 정책망 자체를 업데이트(policy gradient)
  - 평가한 결과에 따라 정책 업뎃
    - **Policy gradient로 정책을 업데이트** : ![](https://latex.codecogs.com/gif.latex?%5Ctheta%27%20%3D%20%5Ctheta%20&plus;%20%5Calpha%20%7B%5Ccolor%7BRed%7D%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta%29%20%7D)
    - ![](https://latex.codecogs.com/gif.latex?%5Ctheta%27%20%3D%20%5Ctheta%20&plus;%20%5Calpha%20%7B%5Ccolor%7BRed%7D%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta%29%20%7D)을 알아내는 접근방법 2가지
      - REINFORCE : return값
      - Actor-Critic : 가치함수
- Policy-Grdient계산
  - 어떤 상태에 에이전트가 있었는지
    - State distribution가 세타값에 따라 달라진다
  - 각 상태에서 에이전트가 어떤 행동을 선택했는가
    - 정책에 따라 확률적으로 행동을 선택함 -> 세타값에 따라 달라진다
- Policy Gradient Theorem
  - ![](https://latex.codecogs.com/gif.latex?%5Cnabla%20J%28%5Ctheta%29%20%3D%20%5Csum_s%20d_%7B%5Cpi_%7B%5Ctheta%7D%7D%28s%29%20%5Csum_a%20%5Cnabla_%7B%5Ctheta%7D%20%5Cpi_%7B%5Ctheta%7D%28a%7Cs%29q_%7B%5Cpi_%7B%5Ctheta%7D%7D%28s%2C%20a%29)
  - ![](https://latex.codecogs.com/gif.latex?%5Cnabla_%7B%5Ctheta%7DJ%28%5Ctheta%29%20%3D%20%5Cnabla_%7B%5Ctheta%7D%20log%20%5Cpi_%7B%5Ctheta%7D%28a%7Cs%29q_%7B%5Cpi_%7B%5Ctheta%7D%7D%28s%2C%20a%29)
  - ![](https://latex.codecogs.com/gif.latex?q_%7B%5Cpi_%7B%5Ctheta%7D%7D%28s%2C%20a%29)를 구하는 방법
    - return값을 사용하는 방법 : REINFORCE
    - critic network를 사용하는 방법 : Actor-Critic
- REINFORCE 알고리즘 순서
  - 한 에피소드를 policy-network에 따라 실행
  - 에피소드를 기록
  - 에피소드가 끝난 뒤 방문한 상태들에 대한 리턴값을 계산
  - ![](https://latex.codecogs.com/gif.latex?%5CSigma_%7Bt%3D0%7D%5E%7BT-1%7D%20%5Cnabla_%7B%5Ctheta%7D%20log%20%5Cpi_%7B%5Ctheta%7D%28A_t%7C%20S_t%29G_t)를 계산하여 정책망을 업데이트함
  - 앞의 4단계를 반복함
- baseline를 사용한 REINFORCE
  - 원인 : Variance이 커지는 문제. 각각의 에피소드가 끝날때가지 기다리기 때문임!!!! 
  - ![](https://latex.codecogs.com/gif.latex?%5Cnable_%7B%5Ctheta%7D%20J%28%5Ctheta%29%20%3D%20%5Csum_s%20d_%7B%5Cpi_%7B%5Ctheta%7D%7D%28s%29%20%5Csum_a%20%5Cnabla_%7B%5Ctheta%7D%20%5Cpi_%7B%5Ctheta%7D%28a%7Cs%29%28q_%7B%5Cpi_%7B%5Ctheta%7D%7D%28s%2C%20a%29%20-%20b%28S_t%29%29)
  - 모형에 적용
  - 가치함수를 베이스라인으로 적용함(가치함수 네트워크를 설정함)
  - 가치함수 네트워크를 업데이트함(TD-error)

### 9. Actor-Critic
- Actor와 Critic
  - Actor
    - 정책을 근사하는 세타값
    - 업데이트 수식
      
