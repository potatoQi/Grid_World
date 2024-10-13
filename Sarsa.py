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
TODO: 请完善函数Sarsa()

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

def policy_improve(state):
    best_action, max_val = -1, -1e10
    for action in a.a_seq:
        a.upd_pi(state, action, 0)
        v = a.action_value(state, action)
        if v > max_val:
            max_val = v
            best_action = action
    a.upd_pi(state, best_action, 1)

def Sarsa():
    for state in a.s_seq:
        a.upd_pi(state, np.random.randint(4 + 1), 1)

    alpha = 0.1

    q_t = copy.deepcopy(a.action_value_tab)
    
    while 1:
        q_k = copy.deepcopy(a.action_value_tab)
        while 1:
            # 算出当前的action value
            for state in a.s_seq:
                for action in a.a_seq:
                    reward = a.get_move_reward(state, action)
                    state_new = a.move(state, action)
                    if a.is_out(*state_new):
                        state_new = state
                    for aa in a.a_seq:
                        if a.pi(state_new, aa) == 1:
                            action_new = aa
                            break
                    q = a.action_value(state, action)
                    qq = a.action_value(state_new, action_new)
                    a.upd_action_value(state, action, q - alpha * (q - (reward + a.gamma * qq)), q_k)
            if a.get_action_value_Gap(q_k, a.action_value_tab) < 1e-5:
                break
            else:
                a.action_value_tab = copy.deepcopy(q_k)
        
        # policy improve
        for state in a.s_seq:
            policy_improve(state)

        # 终止条件
        error = a.get_action_value_Gap(q_t, a.action_value_tab)
        print(error)
        if error < 1e-5:
            break
        else:
            q_t = copy.deepcopy(a.action_value_tab)
            a.push_action_value(online=True)

    a.report()
    a.plot_end_map(Flush=False)
    a.plot_end_convergence()

# ------------------------------------------------
# 不管什么算法，只要是model-free的
# 如果你没有双重循环确保能保证更新到每个state-action pair，而是直接采样一条，那么你的policy更新必须要带有探险属性
# 否则那些没在你策略中的state-action pair将永远不会被访问到（因为首先你没有双重循环，其次采样是按照你policy去采样的，而你的policy又不带有探索，所以采出来的永远是那些pairs）

# 所以要么你就保证采样的同时能强制保证所有state-action pairs被访问到
# 要么就保证你的policy具有探索性，这样robot自己会去采样到其余的pairs

# 我这份代码显然是前者，即强制所有state-action pairs被访问到，并且我没采样，因为我觉得没必要，因为Sarsa本来就是MC Exploring Starts的完全记忆化版本
# 每个状态的更新只依赖下一个状态，如果你采样的话感觉是多余的，只有前两个状态会用到，后面的状态即使我不采样后面由于我双重循环也会访问到

# 当然如果把双重循环去掉，那么我可以使用采样 + 探索性policy，同样可以保证所有pairs都被访问到
# 迭代式跟本程序一样，这也是Sarsa的另一种写法
Sarsa()