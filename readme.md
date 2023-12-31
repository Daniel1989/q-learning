## 概要

### 训练循环
1. environment将state给neural network
2. nn产生q-values，给agent
3. agent产生一个action，给environment

类比下棋，environment就是棋盘，下棋人就是agent，state是当前棋盘的状态。
action就是走哪一步棋，然后看带来了什么reward

https://github.com/HestonCV/rl-gym-from-scratch
https://towardsdatascience.com/develop-your-first-ai-agent-deep-q-learning-375876ee2472#c87e

### 整体流程
1. 初始化一个数组
2. 根据一个state，产生一个action，可能开始是随机的，后面可能是根据model来产生的
3. 根据最新的action，产生next state，以及reward
4. 将这一步加入到经验中
5. 如果有足够经验，则使用模型进行训练
6. 开始下一步
7. 最终产生一个模型，后面可以用模型直接来玩完成一个任务

TODO
1. 去调模型，看5000次有多少步完成的，以及平均多少步。然后后加上模型对比
2. 用完成的模型能否来完成任务
3. 这种训练为什么用sequence这样的模型
