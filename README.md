# Reinforcement Learning
- 학습전 이해 : MP, DP, SP, MDP(https://norman3.github.io/rl/)
- n-step, episode, sequence
- deterministic policy vs stochastic policy
- policy 2가지 형태
  - 정책(상태) -> 명시적(explicit) -> Value Iteration
  - 선택(가치함수(상태)) -> 내재적(implicit) -> Policy Iteration

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
  - evaluation : 벨만기대방성식을 이용한 참 가치함수를 계산, 정책 파이에 대한 참 가치함수를 반복적으로, 모든 상태에 대해 동시에(한번)
  - improvement : 가치함수로 정책 파이를 업데이트, greedy policy
  - SARSA
 
### 4. Reinforcement(고전적) : Model-Free(sampling)
- Off-policy vs On-policy
- **SARSA** : s,a,r,s',a' -> 큐함수 업뎃, 벨만기대방정식
  - 벨만기대방정식을 변형한 큐함수 업데이트
    - ![](http://latex.codecogs.com/gif.latex?q%28s%2C%20a%29%20%3D%20q%28s%2C%20a%29%20&plus;%20%5Calpha%28r%20&plus;%20%5Cgamma%20q%28s%27%2C%20a%27%29%20-%20q%28s%2Ca%29%29)
  - evaluation : TD Learning(Bootstrap), 샘플링으로 대체
  - improvement : 엡실론-탐욕정책(강제 최적이 아닌 행동 선택)
  - 문제점 : on-policy(안좋은 보상을 만날경우 큐함수 업데이트가 지속적 감소)
  
- **Q-Learning** : Value Iteration에 sampling적용, Off-Policy(2개의 정책) -> s,a,r,s'
  - tip) off-policy : behavior policy(샘플수집정책 : 업데이트X), target policy(에이전트의 정책:업데이트o)
  - 벨만최적방정식으로 큐함수 업데이트, off-policy(행동하는 정책, 학습하는 정책)
    - ![](http://latex.codecogs.com/gif.latex?q%28s%2C%20a%29%20%3D%20q%28s%2C%20a%29%20&plus;%20%5Calpha%28r%20&plus;%20%5Cgamma%20%7B%5Ccolor%7BRed%7Dmax%20%7D_%7Ba%27%7D%20q%28s%27%2C%20a%27%29%20-%20q%28s%2Ca%29%29)
  - 행동정책의 종류 : 엡실론탐욕, 볼츠만 등..
- SARSA vs Q-Learning
  - on-policy TD Learning vs off-policy TD Learning
  - Update target :
    - ![](http://latex.codecogs.com/gif.latex?r%20&plus;%20%5Cgamma%20q%28s%27%2C%20a%27%29) vs ![](http://latex.codecogs.com/gif.latex?r%20&plus;%20%5Cgamma%20%7B%5Ccolor%7BRed%7Dmax%20%7D_%7Ba%27%7D%20q%28s%27%2C%20a%27%29)
