# RL 基础
## 概念
*  符号定义
    * 状态 `s`、动作 `a`、奖励 `r`、回报 `u`（奖励 reward 的折现和）。  
    例如，AlphaGo 中 `s_t` 是棋局，`a_t` 是下一步可以落子的位置，`U_t` 是回报。

    * `π_θ(a|s)` 代表策略网络，`θ` 是 `π` 的权重。`Q(s,a)` 是动作价值函数，`V(s)` 是状态价值函数。

* Value-based 和 Policy-Based
   | Value-Based | Policy-Based |
   |-------------|--------------|
   | Q-Learning，DQN  | Policy Gradient、Actor-Critic、DDPG、PPO |
   |通过 `argmax_a Q(s,a)` 来选择动作 | 直接通过 `π(a\|s)` 来选动作  |
   | Q-Learning 不能处理连续动作，DQN 可以处理连续动作但求 `argmax_a` 也不容易 | 处理离散、高维更高效 |

* Value Function 是什么有什么用？
   * Value Function 也用于提供反馈信号。相比 reward 的不同点：When evaluating partial paths, reward focuses on the current states, while value focuses on the unseen future outcomes
      * `reward 一般指当前的反馈，value 可以理解成当前和未来 reward 的期望`
   * 例如：
      * Actor-Critic：同时用到 `π_θ(a|s)` 和 `V_π(s_t)`，其中 `V` 用来评价 `π_θ` 的好坏    
      * MCTS 需要用到哪些网络？一个策略网络 `π_θ(a|s)`、一个状态价值网络 `V_π(s_t)`，`V`用来构建蒙特卡洛树

### 策略网络 Policy Network：π_θ(a|s)
策略网络是根据现有状态 `s`，决定下一步做什么 `a` 的网络。可以通过 behavior cloning 和 policy gradient 进行训练

* behavior cloning 是监督训练，policy gradient 是 RL
* policy gradient 主要想要更新的是参数 `θ`，其梯度有如下近似

    <p align="left" >
    <img src="./pictures/policy_gradient.png" width="600">
    </p>
  
  直观理解，第一项控制 `θ` 更新的方向，第二项是步长。要更新策略网络的参数 `θ`，重要的是得到 `Q_π(s_t,a_t)`，也就是动作价值函数，下面讲

### 价值函数：动作价值和状态价值
> 值函数 `Q_π`和 `V_π` 都可以从经验 Experience 中的很多次模拟中，估计一个出来（蒙特卡罗方法）。但如果状态空间特别大，可以用深度神经网络去实现拟合 Q 或 V，也即 Deep Reinforcement Learning
* 动作价值：`Q_π(s_t,a_t) = E[U_t|s_t, a_t]`
    * 衡量的是当前状态下，每个 action 的回报，可以用观测到的回报 `u_t` 近似替代
    * 例如在 AlphaGo 中，两个策略网络互相博弈，胜出者 `u1 = u2 = ... = u_t = 1`，失败者 `u1 = u2 = ... = u_t = -1`

* 状态价值：`V_π(s_t) = E[U_t|s_t]`
    * 衡量的是当前状态下的胜率。可以用一个神经网络 `v(s;w)` 来近似 `V_π(s)`，其中 `w` 是网络权重


<br>

## 具体的 Q, V, r, R, A 等的公式

### 1. 即时奖励 $r_t$
- **定义**： $r_t$ 是在 **时刻 $t$** 执行某个动作后从环境中获得的 **即时奖励**。它是强化学习中智能体与环境交互的基础，用于反馈智能体当前行为的好坏。
- **例子**：假设你正在玩一个游戏，每当你成功跳跃时，你获得一个奖励 +1；如果失败，则奖励为 0 或 -1。每个时刻获得的奖励就是 $r_t$。

### 2. 状态价值函数 $V(s)$
- **定义**： $V(s)$ 表示在 **某个状态 $s$** 下，智能体 **遵循某个策略** 后，预计能获得的 **未来回报的期望值**。即从状态 $s$ 开始，按照策略 $\pi$ 进行行为，所能获得的总回报。
  
  $$V(s) = \mathbb{E} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s \right]$$

  其中， $\gamma$ 是折扣因子，表示未来奖励的重要性。

- **含义**： $V(s)$ 是一个状态的 **长期价值**，它反映了从当前状态开始执行策略后的 **期望回报**。

- **例子**：在棋类游戏中，状态 $s$ 可以是当前的棋盘配置，$V(s)$ 则表示从该配置开始，按照某种策略（如最优策略）能够获得的未来奖励的预期值。

### 3. 动作价值函数 $Q(s, a)$
- **定义**： $Q(s, a)$ 表示在 **状态 $s$** 下，采取 **动作 $a$** 后，智能体预计能获得的 **未来回报的期望值**。这里 $Q(s, a)$ 不同于 $V(s)$，因为它考虑了在当前状态下选择特定动作后的回报。
  
  $$Q(s, a) = \mathbb{E} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s, a_t = a \right]$$

  -  $Q(s, a)$ 的值是 **通过执行某个动作** 后的回报期望。

- **含义**： $Q(s, a)$ 是 **状态-动作对** 的长期价值，它告诉我们在当前状态下，选择某个动作后，所能获得的总回报。

- **例子**：在棋类游戏中，状态 $s$ 可能是某种棋盘配置，动作 $a$ 是棋手的下一步选择。 $Q(s, a)$ 就表示从这个配置出发，选择该动作后，游戏结束时能获得的回报期望。

### 4. 奖励 $r_t$ vs 总奖励 $R_t$
- **即时奖励 $r_t$**：是当前时刻执行某个动作后，智能体从环境中获得的奖励，通常是一个标量值，表示一个即时反馈。

- **总奖励 $R_t$**：表示从某个时刻 $t$ 开始，智能体在之后的整个过程中所获得的 **累计奖励**。通常它包括了未来多个时刻的奖励，计算公式如下：
  
  $$R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$
  
  这里， $\gamma$ 是折扣因子，表示未来奖励的重要性。

  - **区别**：即时奖励 $r_t$ 是每个时刻收到的奖励，而总奖励 $R_t$ 是从某时刻开始，经过多个时刻累积的奖励。

### 5. 优势函数 $A(s, a)$
- **定义**：优势函数 $A(s, a)$ 反映了在 **状态 $s$** 下，选择 **动作 $a$** 相对于 **平均策略**（通常是基于价值函数的期望回报）的 **优势**。它通常是 **动作价值函数 $Q(s, a)$** 与 **状态价值函数 $V(s)$** 之间的差值：

  $$A(s, a) = Q(s, a) - V(s)$$

- **含义**： $A(s, a)$ 评估了一个动作在当前状态下的 **相对好坏**。它反映了 **当前动作** 相对于 **平均策略** 的 **优越性**。

- **例子**：如果你在棋类游戏中处于某个状态 $s$，并且选择了某个动作 $a$，优势函数 $A(s, a)$ 就告诉你，相比于平均策略，从状态 $s$ 开始，选择动作 $a$ 能带来多少额外的奖励。

### 6. TD误差 $\delta_t$
- **定义**：时间差分误差（**TD误差**）是用于更新状态-价值函数或动作-价值函数的关键量。它度量了 **当前价值估计** 与 **实际观察到的价值** 之间的差异。

  $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

  优势 $A$ 可以用 $\delta$ 来估计（generalized advantage estimation）: $$A_t = \sum_{l=0}^{L-t}(\gamma\lambda)^l \delta_{t+l}$$

  **含义**：TD误差是用来更新价值函数的核心量，它表示 **当前状态价值的估计与下一状态价值的预测误差**。它通过 **即时奖励 $r_t$** 和 **下一状态的价值 $V(s_{t+1})$** 来调整当前状态的价值 $V(s_t)$


### 7. Bellman方程
**Bellman方程** 是描述 **价值函数** 和 **动作价值函数** 的递归方程

- 对于 **状态价值函数** $V(s)$：

  $$V(s_t) = \mathbb{E}[r_t + \gamma V(s_{t+1})]$$

- 对于 **动作价值函数** $Q(s, a)$：

  $$Q(s_t, a_t) = \mathbb{E}[r_t + \gamma \max_{a'} Q(s_{t+1}, a')]$$

  这里， $$\max_{a'} Q(s_{t+1}, a')$$ 表示从下一状态开始，选择最优动作的价值。

### 总结
- **$r_t$** 是 **即时奖励**，反映了当前行为的反馈。
- **$V(s)$** 是 **状态价值函数**，表示从某状态开始，执行某个策略后，能获得的总回报的期望。
- **$Q(s, a)$** 是 **动作价值函数**，表示从某状态出发，执行某个动作后，能获得的总回报的期望。
- **$A(s, a)$** 是 **优势函数**，反映了在某状态下，选择某个动作的 **相对好坏**，即动作的 **优势**。
- **$\delta_t$** 是 **TD误差**，表示当前的价值估计与实际回报之间的差异。


<br>
<br>

# 算法

### MCTS 算法步骤：Selection、Expansion、Simulation、BP
> https://www.youtube.com/watch?v=y2avl6b5ObQ 

<p align="left" >
<img src="./pictures/mcts_2.png" width="600">
</p>

MCTS 算法一共4步：Selection、Expansion、Simulation、BP。整个过程会重复很多次

* Selection：选一个 **假想的** 要探索的动作
    * 例如下图用的是一个传统的 UCB 公式。其中 `N` 是总的探索次数，`n_i` 是当前节点的探索次数，`V` 是节点的 value。会鼓励选择：分数高并且探索次数少的节点。

        <p align="left" >
        <img src="./pictures/mcts.png" width="600">
        </p>

* Expansion
    * 如果 Selection 的节点没被探索过，也即 `n=0`，那么枚举当前节点所有可能的动作，选取子节点为当前节点
* 对 Expansion 得到的子节点进行 Simulation（也叫 Playout/Rollout）
    * AlphaGo 中，会让两个策略网络博弈到游戏结束，分出胜负
* BP，更新蒙特卡洛树上的 `V`、`N`、`n` 等记录


### AlphaGo 中 MCTS 的 4个 步骤
> https://www.youtube.com/watch?v=zHojAp5vkRE&t=172s

* 用监督学习 Behavior Cloning 背棋谱
    * 主要作为 policy network 的初始化（但 behavior cloning 很难学到棋谱外的棋局）

* 训练策略网络 `π_θ(a|s)`
    * 让两个 policy 网络互相博弈，直到分出胜负（self-play）
    * 得到 trajectory：`s1，a1，s2，a2，...，S_t，a_t`
    * 根据胜负，更新 player 的策略网络。而 opponent 不更新，从过去的策略网络中随机选一个即可

        <p align="left" >
        <img src="./pictures/policy_gradient2.png" width="600">
        </p>
    * 因为目标是最大化回报，所以用的是梯度上升

* 训练价值网络
    * 也是让两个策略网络博弈，目的是最小化策略网络 `v` 的估计和 `u` 的偏差，所以用梯度下降
        <p align="left" >
        <img src="./pictures/value_gradient.png" width="500">
        </p>
    * 其中策略网络和价值网络可以用相同的主干网络得到，但用不同的 head。对于围棋（19*19），策略网络的输出是 361 维的向量，价值网络是标量。

* 在实战推理的时候用 MCTS（策略网络和价值网络都训练好了，只更新 MCTS 树中的值）

    * 其 selection 用的是如下公式
        <p align="left" >
        <img src="./pictures/mcts_alphago.png" width="500">
        </p>
    * Simulation：AlphaGo 中，会让两个策略网络博弈到游戏结束。在这个过程中，用状态价值网络 `v(s;w)` 对每个状态打个分，最终的分数是也会考虑进胜负的奖励（±1），最终 `V(s_t+1) = 0.5*v(s_t+1; w) + 0.5*r_t`
    * 每次 BP，AlphaGo 会利用每个状态下，状态价值网络的分数平均值 `mean(the record of V's）`，更新第一步 selection 中每个节点的 `Q(a_t)`


### AlphaZero 
* AlphaGo 为什么要 MCTS？---如果只用策略网络，可能一个小错误就会导致出现大的偏差
* AlphaZero 和 AlphaGo 的区别？--- 策略网络训练没有用 Behavior Cloning，而是用了 MCTS，如下图

    <p align="left" >
    <img src="./pictures/alphazero.png" width="500">
    </p>

<br>


## LLM 中的 RL
核心是（训练）得到一个能够给出评分的 `reward model / verifier`，对 policy model 的 solution 给出评分

其中，**PPO 是在 post-training 阶段用 reward model，主要是用于对齐人类偏好；
MCTS 等是在 inference 阶段用 reward model，结合 LLM searching（通过提出很多猜测+验证，提升模型能力）**

### Post-training 阶段：RLHF（PPO）, DPO
> [Direct Preference Optimization (DPO) for LLM Alignment (From Scratch)](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
* PPO 会先训练一个 reward model，用 reward model 去训练 policy 网络
* DPO：作为 PPO 的简化，直接利用偏好数据训练 LLM Policy 网络
     <p align="left" >
     <img src="./pictures/dpo.png" width="700">
     </p>

### 推理阶段使用 reward model
#### MCTS
* 概念
   * `策略网络 policy networks` 也即最终要回答问题的大模型
   * `价值网络 value function` 用于在 MCTS 中给出相应分数（作为 critic）
       * 用策略网络 LLM 的权重初始化，加一个 MLP 层，尝试最小化基于每个 s 去预测 return u 的偏差
   * 还有两个经常遇见的概念，作为 value function 的补充：`ORM（Outcome Reward Model），PRM（Process Reward Model`。他们功能和价值网络类似，分别是给出稀疏和稠密的 immediate reward
       * ORM：用于给出  a sequence of actions or options `o_{1:T}` 的整体的分数，代表整个 sequence 的成败或质量
       * PRM helps predicts the immediate action-specific reward given the state and the option (可以理解为动作): `R(s_t, o_t)`

* Paper：Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing 文章
    * 训练 `价值网络 value function`，还有 ORM，PRM 等，用于在 MCTS 中给出相应分数（作为 critic）
    * 用 MCTS 去构建更好的 trajectory（也即更高质量的数据）
    * 用新构建的数据对 `策略网络 policy networks` 进行 SFT

    <p align="left" >
    <img src="./pictures/llm_mcts.png" width="600">
    </p>

* `Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B`：全程让大模型自己给 reward，构建 MCTS 树

#### Outcome Value Models + BeamSearch
> OVM, Outcome-supervised Value Models for Planning
in Mathematical Reasoning
* 推理前：用 `mean squared error + 构建的 (question, solution, binary label) 数据集（N个问题，每个问题 n 个path）`，去微调 LLM 得到 OVM（Outcome Value Models）
* 推理时：对于 top-k 的 beamsearch，也考虑进 OVM 的分数，选取 top-b 的 path。最终的 final answer 选取 final value 最高的 path

#### Verifier + CoT
> Generative Verifiers: Reward Modeling as Next-Token Prediction
* 推理前：训练一个能对 CoT 过程进行 yes/no 的 verifier
* 推理时：At test-time, we sample multiple CoT rationales and use `majority voting` to compute the average probability of ‘Yes’
