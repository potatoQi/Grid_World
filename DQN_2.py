from grid_world import GRID_WORLD
import copy
import numpy as np
import time
import sys

# 实例化
a = GRID_WORLD(
    random_seed = 42,           # 随机种子（生成网格世界）
    n = 10,                     # 网格世界的长度
    m = 10,                     # 网格世界的宽度
    bar_ratio = 0.3,            # 障碍的比例
    r_bar = -100,               # 障碍的reward
    r_out = -1,                 # 碰到边界的reward
    r_end = 10,                 # 达到终点的reward
    autocal = 1,                # 是否使用内置算法(值迭代)计算最优policy, state values, action values
    gamma = 0.9,                # 值迭代算法的gamma
    eps = 1e-5,                 # 值迭代算法的收敛条件
)

'''
TODO: 请完善函数DQN_2()

关于接口：
    1. 环境对象已实例化为'a', 实例化参数可更改，请合理利用接口, 可用的接口如下:
        0)   属性, a.gamma:                              γ
        1)   属性, a.s_seq:                              状态的集合
        2)   属性, a.a_seq:                              动作的集合
        3)   属性, a.r_seq:                              reward的集合
        4)   属性, a.pi_tav:                             π(a | s)
        5)   属性, a.state_value_tab:                    v(s)
        6)   属性, a.action_value_tab:                   q(s, a)
        7)   方法, a.prob_reward(s, a, r):               p(r | s, a)
        8)   方法, a.prob_state(s, a, s'):               p(s' | s, a)
        9)   方法, a.pi(s, a, [obj]):                    读取π(a | s)
        10)  方法, a.state_value(s, [obj]):              读取v(s)
        11)  方法, a.action_value(s, a, [obj]):          读取q(s, a)
        12)  方法, a.upd_pi(s, a, val, [obj]):           修改π(a | s)为val
        13)  方法, a.upd_state_value(s, val, [obj]):     修改v(s)为val
        14)  方法, a.upd_action_value(s, a, val, [obj]): 修改q(s, a)为val
        15)  方法, a.get_move_reward(s, a):              返回在s执行a后的reward
        16)  方法, a.get_action_value_Gap(q1, q2):       返回俩action value之间的差距

    2. 您与环境交互的变量为: a.pi_tab, a.state_value_tab, a.action_value_tab。

    3. 上述的读取和修改接口默认是连通着上面三个量, 但是可选参数obj可指定为自己创建的字典, 可帮助您降低编码难度。

关于测评：
    1. 若要测评, 请务必实例化时打开autocal参数

    2. 当您的algorithm正文结束后, 运行a.plot_end_map()可视化algorithm的最终policy

    3. 若想查看algorithm的收敛图, 请在algorithm每一次迭代后加上语句a.push_state_value([online]), online=True将开启实时画图
        并在algorithm正文结束后, 运行a.plot_end_convergence()可视化algorithm的收敛情况

    4. 若想查看algorithm的文字报告, 当您的algorithm正文结束后, 运行a.report()
'''

import torch
import torch.nn as nn  # 全连接层
import torch.nn.functional as F

class QNetwork(nn.Module):  # 定义Q网络类，继承自nn.Module
    def __init__(self, state_size=2, action_size=5, hidden_size=64):
        """
        初始化Q网络的结构。
        :param state_size: 状态空间的维度，即输入的特征数量
        :param action_size: 动作空间的维度, 即输出动作Q值的数量
        :param hidden_size: 隐藏层的大小(神经元个数), 默认为64
        """
        # 声明一个子类前必须声明它的父类
        super(QNetwork, self).__init__()
        # 第一层全连接层：将状态输入（state_size维度）映射到隐藏层（hidden_size维度）
        self.fc1 = nn.Linear(state_size, hidden_size)
        # 第二层全连接层：将隐藏层进一步映射到另一个隐藏层（hidden_size维度）
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 第三层全连接层：将隐藏层映射到动作空间（action_size维度），输出每个动作的Q值
        self.fc3 = nn.Linear(hidden_size, action_size)
    def forward(self, state):
        """
        前向传播函数，定义网络的计算流程。
        :param state: 输入状态(batch_size, state_size)，即输入的状态向量
        :return: 每个动作的Q值(batch_size, action_size)
        """
        # 将输入状态通过第一层全连接层，并应用ReLU激活函数
        x = torch.relu(self.fc1(state))
        # 将第一层的输出通过第二层全连接层，并应用ReLU激活函数
        x = torch.relu(self.fc2(x))
        # 将第二层的输出通过第三层全连接层，不使用激活函数，得到每个动作的Q值
        q_values = self.fc3(x)
        return q_values  # 返回动作的Q值

import collections  # 数据结构
import random

class Buffer:
    def __init__(self, maxLength=10000):
        self.buffer = collections.deque(maxlen=maxLength)
    def add(self, state, action, reward, state_new, done):
        self.buffer.append((state, action, reward, state_new, done))
    def sample(self, batch_size=10):
        transitions = random.sample(self.buffer, batch_size)
        # zip是把每个参数的第一个元素们打包成一个元组，第二个元素们打包成一个元组... ...
        state, action, reward, state_new, done = zip(*transitions)
        return state, action, reward, state_new, done
    def size(self):
        return len(self.buffer)

class DQN_2():
    def __init__(self, state_dim=2, action_dim=5, hidden_dim=64, learning_rate=0.001, gamma=0.9, epsilon=0.3, tim_of_net_upd=10, device="cpu", buffer_max_length=10000, buffer_start_size=500, batch_size=64):
        # robot的基本属性
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        # robot内置算法属性
        self.buffer = Buffer(maxLength=buffer_max_length)
        self.buffer_start_size = buffer_start_size
        self.batch_size = batch_size
        self.q_net_1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)  # net1是固定住的, 即用来算r + γ * max q
        self.q_net_2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)  # net2是时刻更新的, 用来拟合r + γ * max q
        # 可以理解为net1是目标，然后net2去追赶它，等追赶到了那么net1就更新为net2
        self.optimizer = torch.optim.Adam(self.q_net_2.parameters(), lr=learning_rate)
        self.cnt = 0
        self.tim_of_net_upd = tim_of_net_upd
        self.device = device

    def state_to_tensor(self, state):
        return torch.tensor([list(state)], dtype=torch.float).to(self.device)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # 这里按照net2的策略去走是因为此时你在更新net2, 也就是在追赶net1的过程中, 所以当然是采net2的样
            action = self.q_net_2(self.state_to_tensor(state)).argmax().item()
        return action
    
    def update(self, states, actions, rewards, states_new, dones):
        # 获得predict
        states = torch.tensor(states, dtype=torch.float).reshape(-1, 2).to(self.device)     # 排成一列
        actions = torch.tensor(actions, dtype=torch.int64).reshape(-1, 1).to(self.device)   # 排成一列
        predict = self.q_net_2(states).gather(1, actions)
        # 获得label
        states_new = torch.tensor(states_new, dtype=torch.float).reshape(-1, 2).to(self.device) # 排成一列
        qq = self.q_net_1(states_new).max(1)[0].reshape(-1, 1)                  # 排成一列
        rewards = torch.tensor(rewards, dtype=torch.float).reshape(-1, 1).to(self.device)       # 排成一列
        dones = torch.tensor(dones, dtype=torch.int).reshape(-1, 1).to(self.device)             # 排成一列
        label = rewards + a.gamma * (1 - dones) * qq
        # 计算loss
        loss = torch.mean(F.mse_loss(predict, label))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cnt += 1
        if self.cnt % self.tim_of_net_upd == 0:
            self.cnt = 0
            self.q_net_1.load_state_dict(self.q_net_2.state_dict())  # 更新net1
            # 更新action_value_tab & policy
            for state in a.s_seq:
                with torch.no_grad():
                    seq = self.q_net_1(self.state_to_tensor(state))[0]
                best_action, max_val = -1, -1e10
                for action in a.a_seq:
                    val = seq[action].item()
                    a.upd_action_value(state, action, val)
                    a.upd_pi(state, action, 0)
                    if val > max_val:
                        max_val = val
                        best_action = action
                a.upd_pi(state, best_action, 1)
            a.push_action_value(online=True)
            error = a.get_action_value_Gap(a.action_value_tab, a.true_action_value_tab)
            print(error)

    def do_sampling(self, maxLength=1000):
        state = (a.sx, a.sy)
        done = 0
        for i in range(maxLength):
            action = self.take_action(state)
            reward = a.get_move_reward(state, action)
            state_new = a.move(state, action)
            if a.is_out(*state_new): state_new = state
            if i == maxLength - 1:
                done = 1
            self.buffer.add(state, action, reward, state_new, done)
            if self.buffer.size() > self.buffer_start_size:
                states, actions, rewards, states_new, dones = self.buffer.sample(self.batch_size)
                self.update(states, actions, rewards, states_new, dones)
            state = state_new

# ------------------------------------------------------------------
agent = DQN_2(
    # agent基本配置
    state_dim=2,
    action_dim=5,
    gamma=0.9,
    epsilon=1,
    # DQN算法配置
    buffer_max_length=10000,
    buffer_start_size=500,
    batch_size=256,
    hidden_dim=128,
    learning_rate=0.001,
    tim_of_net_upd=1000,
    device="cuda"
)

num_episodes = 200
for i in range(num_episodes):
    agent.do_sampling(maxLength=10000)
a.plot_end_map(Flush=False)

# 我个人觉得从DQN开始，我们就要转变对“探索率”这个的看法了
# 因为此时我们的目标转为了拟合一个函数，怎么样才叫拟合效果好呢？
# 就是任意pair，其值都跟真实值差不多
# 所以此时我们希望覆盖尽可能多的pair，所以此时我们采样的目的不像以前那样是为了估算跟policy有关的一个值
# 而是提供尽可能多丰富的“数据”，即(s, a, r, s')
# 所以从DQN开始，若用DL的方法去做，那么在model-free的情况下，就应该探索性拉满，即机器人的目的就是为了走到不同的地方采数据，而不是为了估计某个值

# 在DQN_1.py中，我用了双重循环，所以保证了每个pair的数据都提供到了。所以最好能收敛到error为1
# 但是因为随机采样的原因，必然有些pair提供的数据有多次，有些pair提供的数据不多，所以它们被拟合的重要性不同，所以error收敛到30就极限了

# 这个版本的写法具有开创意义，第一次实现了agent和environment的封装
