from grid_world import GRID_WORLD
import copy
import sys
import numpy as np
import time

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
TODO: 请完善函数policy_iteration()

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

    2. 您与环境交互的变量为: a.pi_tab, a.state_value_tab, a.action_value_tab。

    3. 上述的读取和修改接口默认是连通着上面三个量, 但是可选参数obj可指定为自己创建的字典, 可帮助您降低编码难度。

关于测评：
    1. 若要测评, 请务必实例化时打开autocal参数

    2. 当您的algorithm正文结束后, 运行a.plot_end_map()可视化algorithm的最终policy

    3. 若想查看algorithm的收敛图, 请在algorithm每一次迭代后加上语句a.push_state_value([online]), online=True将开启实时画图
        并在algorithm正文结束后, 运行a.plot_end_convergence()可视化algorithm的收敛情况

    4. 若想查看algorithm的文字报告, 当您的algorithm正文结束后, 运行a.report()
'''

def policy_iteration():
    start_time = time.time()  # 记录开始时间

    # 定义下一个时刻的policy
    pi_k = copy.deepcopy(a.pi_tab)

    # 当前时刻刚开始时的state value
    v_tmp = copy.deepcopy(a.state_value_tab)

    # 初始化当前时刻的policy（易错：这里不初始化的话state_value_tab更新不下去）
    for state in a.s_seq:
        t = np.random.randint(0, 4 + 1)
        a.upd_pi(state, t, 1)

    while True:
        # 当前的policy为a.pi_tab, state value为a.state_value_tab
        # 那么要基于这个policy迭代算出state value
        v_k = copy.deepcopy(a.state_value_tab)
        while 1:
            for state in a.s_seq:
                sum1 = 0
                for action in a.a_seq:
                    sum2_1, sum2_2 = 0, 0
                    for r in a.r_seq:
                        sum2_1 += r * a.prob_reward(state, action, r)
                    for ss in a.s_seq:
                        sum2_2 += a.prob_state(state, action, ss) * a.state_value(ss)
                    sum1 += a.pi(state, action) * (sum2_1 + a.gamma * sum2_2)
                a.upd_state_value(state, sum1, v_k)

            if a.get_state_value_Gap(a.state_value_tab, v_k) < 1e-5:
                break
            else:
                a.state_value_tab = copy.deepcopy(v_k)

        # 算出基于此时a.state_value_tab的action value，同时在线更新policy
        # 经实测，这里在线或者离线更新policy都可以收敛
        for state in a.s_seq:
            for action in a.a_seq:
                sum1 = 0
                for r in a.r_seq:
                    sum1 += r * a.prob_reward(state, action, r)
                sum2 = 0
                for ss in a.s_seq:
                    sum2 += a.prob_state(state, action, ss) * a.state_value(ss)
                a.upd_action_value(state, action, sum1 + sum2 * a.gamma)
            # 计算完一组q(s, a)就更新下π(s, a)
            best_action = -1
            max_val = -1e10
            for action in a.a_seq:
                a.upd_pi(state, action, 0, pi_k)
                v = a.action_value(state, action)
                if v > max_val:
                    max_val = v
                    best_action = action
            a.upd_pi(state, best_action, 1, pi_k)

        # 结束条件
        if a.get_state_value_Gap(v_tmp, a.state_value_tab) < 1e-5:
            break
        else:
            v_tmp = copy.deepcopy(a.state_value_tab)
            a.pi_tab = copy.deepcopy(pi_k)
            a.push_state_value(online=True)

    # check自己的算法
    end_time = time.time()  # 记录结束时间
    print(f"算法运行时间：{end_time - start_time}秒")   
    a.report()
    a.plot_end_map()
    a.plot_end_convergence()

# ------------------------------
policy_iteration()