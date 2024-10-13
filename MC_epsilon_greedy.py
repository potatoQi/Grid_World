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
    autocal = 0,                # 是否使用内置算法(值迭代)计算最优policy, state values, action values
    gamma = 0.9,                # 值迭代算法的gamma
    eps = 1e-5,                 # 值迭代算法的收敛条件
)

'''
TODO: 请完善函数MC_epsilon_greedy()

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

def get_action(state):
    prob = [a.pi(state, i) for i in range(5)]
    elements = [i for i in range(5)]
    res = np.random.choice(elements, p=prob)
    return res

def sampling(state, action, maxLength=100000):
    pipe = [(state, action, a.get_move_reward(state, action))]
    for i in range(maxLength):
        state_new = a.move(state, action)
        if a.is_out(*state_new):
            state_new = state
        action_new = get_action(state_new)
        reward_new = a.get_move_reward(state_new, action_new)
        pipe.append((state_new, action_new, reward_new))
        state, action = state_new, action_new
    return pipe

def policy_improve(state, epsilon):
    best_action, max_val = -1, -1e10
    for action in a.a_seq:
        val = a.action_value(state, action)
        if val > max_val:
            max_val = val
            best_action = action
    for action in a.a_seq:
        if action == best_action:
            a.upd_pi(state, action, 1. - 4. / 5. * epsilon)
        else:
            a.upd_pi(state, action, 1. / 5. * epsilon)

def MC_epsilon_greedy():
    state0 = (a.sx, a.sy)
    for state in a.s_seq:
        for i in range(5):
            a.upd_pi(state, i, 1. / 5.)

    q_s = {}
    cnt = {}
    for state in a.s_seq:
        for action in a.a_seq:
            q_s[state, action] = 0
            cnt[state, action] = 0

    q_t = copy.deepcopy(a.action_value_tab)
    epsilon = 1
    iter_num = 200

    for iter in range(iter_num):
        for state in a.s_seq:
            for action in a.a_seq:
                q_s[state, action] = 0
                cnt[state, action] = 0

        action0 = get_action(state0)
        pipe = sampling(state0, action0, maxLength=100000)
        sum = 0
        for item in reversed(pipe):
            ss, aa, rr = item
            sum = rr + sum * a.gamma
            q_s[ss, aa] += sum
            cnt[ss, aa] += 1
            a.upd_action_value(ss, aa, q_s[ss, aa] / cnt[ss, aa])

        for state in a.s_seq:
            policy_improve(state, epsilon=epsilon)
        
        error = a.get_action_value_Gap(a.action_value_tab, q_t)
        print(error, epsilon, iter)
        if error < 1e-2:
            break
        else:
            q_t = copy.deepcopy(a.action_value_tab)
            a.push_action_value(online=True)
            if epsilon > 0.0:
                epsilon -= 0.01
                if epsilon < 0:
                    break
    
    a.plot_end_map(Flush=False)

# ------------------------------------------------
# 这个算法能一步到达最佳policy很难
# 因为它带有概率，所以无论如何，在采样过程中有概率不会按照最优policy走，导致不优的q(s,a)变高，从而下一轮不优的q反而变最优了，从而最优policy改变
# 所以这个算法的优点就是可以让robot从一个点自己去探索，探索出个差不多的policy
# 所以得到差不多的policy之后，还要搭配不带贪心概率的算法再跑一下
MC_epsilon_greedy()