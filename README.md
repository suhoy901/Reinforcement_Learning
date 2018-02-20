# Reinforcement_Learning : 
- 목적 : accumulative future reward을 최대화하는 policy를 찾아내는 것
## Index
### 1. Markov Decision Process
- MDP : State, Action, State transition probability, Reward, Discount_Factor, Policy
- State Value Function(상태가치함수) : Q-function기대값이며 policy사용, 반환값(보상,감가율), 상태
- Q-function(행동가치함수) : 특정 state에서 a라는 action을 취할 경우의 받을 return에 대한 기대값 

### 2. Bellman Equation
- Bellman Expectation Equation(벨만기대방정식) : 특정 정책에 대한 가치함수
  - Backup(dynamic programming, reinforcement learning)
- Bellman Optimality Equation : optimal value function사이의 관계식 ---> DynamicProgramming
  - Optimal value function : environment에서 취할 수 있는 가장 높은 값의 reward 총합(Deterministic)
  
### 3. Dynamic Programming(full-width backup) : Model-Base
 - 현재 optimal하지 않는 어떤 policy에 대해서 value function을 구하고(prediction) 현재의 value function을 토대로 더 나은 policy를 구하고 이와 같은 과정을 반복하여 optimal policy를 구하는 것
 - Policy Iteration(Bellman Expectation Equation) : 정책평가 + 정책발전, GPI
   - evaluation : 정책 파이에 대한 참 가치함수를 반복적으로, 모든 상태에 대해 동시에(한번)
   - improvement : 가치함수로 정책 파이를 업데이트, greedy
 - Value Iteration(Bellman Optimality Equation)
 
### 4. Reinforcement(고전적) : Model-Free
- SARSA : s,a,r,s',a' -> 큐함수 업뎃, 벨만기대방정식
  - evaluation : TD Learning(Bootstrap), 샘플링으로 대체
  - improvement : 엡실론-탐욕정책(강제 최적이 아닌 행동 선택)
  - 문제점 : on-policy(안좋은 보상을 만날경우 큐함수 업데이트가 지속적 감소)
- Q-Learning : Value Iteration에 sampling적용, Off-Policy(2개의 정책) -> s,a,r,s'
  - tip) off-policy : behavior policy(샘플수집정책 : 업데이트X), target policy(에이전트의 정책:업데이트o)
  - 벨만최적방정식으로 큐함수 업데이트, off-policy(행동하는 정책, 학습하는 정책)
  - 행동정책의 종류 : 엡실론탐욕, 볼츠만 등..
  
  


 
