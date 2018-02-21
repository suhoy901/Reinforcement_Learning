# Reinforcement Learning
- 학습전 이해 : MP, DP, SP, MDP(https://norman3.github.io/rl/)
- n-step, episode, sequence
- deterministic policy vs stochastic policy
- policy 2가지 형태
  - 정책(상태) -> 명시적(explicit)
  - 선택(가치함수(상태)) -> 내재적(implicit)

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
 - 현재 optimal하지 않는 어떤 policy에 대해서 value function을 구하고(prediction) 현재의 value function을 토대로 더 나은 policy를 구하고 이와 같은 과정을 반복하여 optimal policy를 구하는 것
 - Policy Iteration(Bellman Expectation Equation) : 정책평가 + 정책발전, GPI
   - evaluation : 정책 파이에 대한 참 가치함수를 반복적으로, 모든 상태에 대해 동시에(한번)
   - improvement : 가치함수로 정책 파이를 업데이트, greedy
 - Value Iteration(Bellman Optimality Equation)
 
### 4. Reinforcement(고전적) : Model-Free
- Off-policy vs On-policy
- SARSA : s,a,r,s',a' -> 큐함수 업뎃, 벨만기대방정식
  - evaluation : TD Learning(Bootstrap), 샘플링으로 대체
  - improvement : 엡실론-탐욕정책(강제 최적이 아닌 행동 선택)
  - 문제점 : on-policy(안좋은 보상을 만날경우 큐함수 업데이트가 지속적 감소)
- Q-Learning : Value Iteration에 sampling적용, Off-Policy(2개의 정책) -> s,a,r,s'
  - tip) off-policy : behavior policy(샘플수집정책 : 업데이트X), target policy(에이전트의 정책:업데이트o)
  - 벨만최적방정식으로 큐함수 업데이트, off-policy(행동하는 정책, 학습하는 정책)
  - 행동정책의 종류 : 엡실론탐욕, 볼츠만 등..
  


 
