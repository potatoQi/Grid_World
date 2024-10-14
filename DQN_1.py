from grid_world import GRID_WORLD
import copy
import numpy as np
import time
import sys

# 实例化
a = GRID_WORLD(
    random_seed = 42,           # 随机种子（生成网格世界）
    n = 5,                     # 网格世界的长度
    m = 5,                     # 网格世界的宽度
    bar_ratio = 0.3,            # 障碍的比例
    r_bar = -100,               # 障碍的reward
    r_out = -1,                 # 碰到边界的reward
    r_end = 10,                 # 达到终点的reward
    autocal = 1,                # 是否使用内置算法(值迭代)计算最优policy, state values, action values
    gamma = 0.9,                # 值迭代算法的gamma
    eps = 1e-5,                 # 值迭代算法的收敛条件
)

'''
TODO: 请完善函数DQN_1()

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
import torch.optim as optim  # 引入PyTorch的优化器模块

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

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

def state_to_tensor(state):
    return torch.tensor([list(state)], dtype=torch.float32).to(device)

def policy_improve(state):
    best_action, max_val = -1, -1e10
    for action in a.a_seq:
        a.upd_pi(state, action, 0)
        val = a.action_value(state, action)
        if val > max_val:
            max_val = val
            best_action = action
    a.upd_pi(state, best_action, 1)

def DQN_1():
    print(device)
    for state in a.s_seq:
        a.upd_pi(state, np.random.randint(5), 1)

    q_k = copy.deepcopy(a.action_value_tab)

    q_network = QNetwork(state_size=2, action_size=5).to(device) # 网络定义
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)     # 优化器定义
    loss_fn = nn.MSELoss()                                       # 损失函数定义

    while 1:
        label_batch = torch.empty(0, 5, dtype=torch.float32).to(device) # 0个样本, 每个样本5个状态
        for state in a.s_seq:
            label = []
            for action in a.a_seq:
                reward = a.get_move_reward(state, action)
                state_new = a.move(state, action)
                if a.is_out(*state_new): state_new = state

                q_predict = q_network(state_to_tensor(state_new))
                max_value = torch.max(q_predict)
                label.append(reward + a.gamma * max_value)

            label = torch.tensor(label).reshape(1, 5).to(device)
            label_batch = torch.cat((label_batch, label))
        
        # 参数更新
        # 就目前写代码而言，本人目前感受到的就是，神经网络的在线程度越高越难收敛
        # 也就是你的label，最好是由一套参数得到的，不要一边更新你的参数，一边得到label
        # 目前我这样的写法相当于用一套参数得到label，然后朝着label方向更新参数1000次，再进行下一轮
        # 即使这样，在0.001的学习率下，收敛速度很慢且最多MSE收敛到1
        # 所以，要想收敛，最好先把当前label学的差不多了，再去更新参数重新算新的label
        for epoch in range(1000):
            predict_batch = torch.empty(0, 5, dtype=torch.float32).to(device)
            for state in a.s_seq:
                predict = q_network(state_to_tensor(state))
                predict_batch = torch.cat((predict_batch, predict))
            loss = loss_fn(predict_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 得到当前参数下的action value
        for state in a.s_seq:
            with torch.no_grad():
                seq = q_network(state_to_tensor(state))[0]
            for action in a.a_seq:
                a.upd_action_value(state, action, seq[action].item())

        # policy improve
        for state in a.s_seq:
            policy_improve(state)
    
        # 终止条件
        error = a.get_action_value_Gap(a.action_value_tab, q_k)
        error_true = a.get_action_value_Gap(a.true_action_value_tab, q_k)
        print(error_true)
        if error < 1e-5:
            break
        else:
            q_k = copy.deepcopy(a.action_value_tab)
            a.push_action_value(online=True)

    a.report()
    a.plot_end_map(Flush=False)
    a.plot_end_convergence()

# ------------------------------------------------------------------
DQN_1()
'''
# 使用案例:

q_network = QNetwork(state_size=2, action_size=5)  # 初始化Q网络

state = torch.randn(1, 2)  # 随机生成1个样本，每个样本特征为2

q_values = q_network(state) # 前向传播，这样使用等于调用forward函数

optimizer = optim.Adam(q_network.parameters(), lr=0.001) # 创建一个优化器，使用Adam优化器更新Q网络的参数，学习率设置为0.001

target_q_values = torch.tensor([[1., 2., 3., 4., 5.]]) # 假设我们已经通过贝尔曼方程计算好了目标Q值（用于训练），这里人为构造一个例子

loss_fn = nn.MSELoss() # 定义损失函数为均方误差损失函数（MSELoss），用于计算预测Q值与目标Q值之间的差距

predicted_q_values = q_network(state) # 通过Q网络对当前状态进行前向传播，得到预测的Q值

loss = loss_fn(predicted_q_values, target_q_values) # 计算损失：比较预测的Q值与目标Q值之间的差异

optimizer.zero_grad() # 在进行反向传播之前，先将优化器的梯度缓存清零
loss.backward() # 反向传播，计算梯度

optimizer.step() # 更新网络的参数
'''