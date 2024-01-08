## 概要

### 训练循环
1. environment将state给neural network
2. nn产生q-values，给agent
3. agent产生一个action，给environment

类比下棋，environment就是棋盘，下棋人就是agent，state是当前棋盘的状态。
action就是走哪一步棋，然后看带来了什么reward

https://github.com/HestonCV/rl-gym-from-scratch
优化能到10个馆子
https://github.com/Zeta36/chess-alpha-zero
学习
https://towardsdatascience.com/develop-your-first-ai-agent-deep-q-learning-375876ee2472#c87e

### 整体流程
1. 初始化一个数组
2. 根据一个state，产生一个action，可能开始是随机的，后面可能是根据model来产生的
3. 根据最新的action，产生next state，以及reward
4. 将这一步加入到经验中
5. 如果有足够经验，则使用模型进行训练
6. 开始下一步
7. 最终产生一个模型，后面可以用模型直接来玩完成一个任务

### 疑问
1. 去调模型，看5000次有多少步完成的，以及平均多少步。然后后加上模型对比
   1. 添加图形，对比加载模型和不加载模型的变化--确实用模型比较好
   2. 打印每次训练是哪个大迭代中哪个步骤，打印训练的长度，打印训练开始时间和结束时间 -- 当部署达到32步后就会进入到训练
   3. 切换gpu，看训练时间是否有提起 -- 不会，因为数据量太小，还不如cpu。具体参考其他
2. 为什么需要在getAction中，偶尔会随机返回 -- 就是为了避免限于局部优点，在早期让agent多多探索，随着后面学习越来越熟练，随机action就越来越少了
3. 看训练的state和target value分别是什么 -- 就是当前的状态和对应的action的值，真正的action从最大值取
4.【TODO】结合游戏

### 其他
1. 关于为什么在小数据下，gpu不如cpu快。参考 https://github.com/pytorch/pytorch/issues/77799