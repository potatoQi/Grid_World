import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

class GRID_WORLD:
    def __init__(self, n=5, m=5, bar_ratio=0.2, r_bar=-10, r_out=-1, r_end=1, random_seed=None, autocal=True, gamma=0.9, eps=1e-5):
        self.n = n
        self.m = m
        self.bar_ratio = bar_ratio
        self.r_bar = r_bar
        self.r_out = r_out
        self.r_end = r_end
        self.r_seq = [0, r_bar, r_out, r_end]
        self.s_seq = [(i, j) for i in range(n) for j in range(m)]
        self.a_seq = [i for i in range(5)]
        self.random_seed = random_seed
        self.prob_reward_tab = {}       # p(r|s,a) （准确计算出来的）
        self.prob_state_tab = {}        # p(s'|s,a)（准确计算出来的）

        self.true_pi_tab = {}           # π(a|s)   （存准确计算出的，self.cal_pi_state_and_action_value()执行完后才有值）
        self.true_state_value_tab = {}  # v_π(s)   （存准确计算出的，self.cal_pi_state_and_action_value()执行完后才有值）
        self.true_action_value_tab = {} # q_π(s,a) （存准确计算出的，self.cal_pi_state_and_action_value()执行完后才有值）

        self.pi_tab = {}                # π(a|s)   （存algorithm估计的）
        self.state_value_tab = {}       # v_π(s)   （存algorithm估计的）
        self.action_value_tab = {}      # q_π(s,a) （存algorithm估计的）

        self.generate_grid_world()
        self.gamma = gamma
        self.eps = eps
        if autocal == True:
            self.cal_pi_state_and_action_value()

        self.convergence = []   # 存algorithm的state values与标准state values之间的MSE

        self.fig, self.ax = plt.subplots()  # 创建图层、坐标轴
        self.ax.set_aspect('equal')  # x, y轴保持一样的比例
    
    def generate_grid_world(self):
        n = self.n
        m = self.m
        bar_ratio = self.bar_ratio
        r_bar = self.r_bar
        r_end = self.r_end
        random_seed = self.random_seed

        if random_seed != None:
            np.random.seed(random_seed)

        # 地图
        a = [[0 for j in range(m)] for i in range(n)]
        a = np.array(a)

        # 生成barriers
        num_bar = int(n * m * bar_ratio)
        vis = [[0 for j in range(m)] for i in range(n)]
        cnt = 0
        while (cnt != num_bar):
            x = np.random.randint(0, n)
            y = np.random.randint(0,m)
            if not vis[x][y]:
                vis[x][y] = 1
                a[x, y] = -1
                cnt += 1

        # 生成起点
        while True:
            x = np.random.randint(0, n)
            y = np.random.randint(0,m)
            if a[x, y] != -1:
                x_str = x
                y_str = y
                a[x, y] = 1
                break

        # 生成终点
        while True:
            x = np.random.randint(0, n)
            y = np.random.randint(0,m)
            if a[x, y] != -1 and (x != x_str or y != y_str):
                if abs(x - x_str) + abs(y - y_str) > (n + m) / 3: # 保证起点和终点之间有一定距离
                    x_end = x
                    y_end = y
                    a[x, y] = 2
                    break
        
        # 生成reward
        r = a.copy()
        for i in range(n):
            for j in range(m):
                if a[i, j] == -1: r[i, j] = r_bar
                elif a[i, j] == 1: r[i, j] = 0
                elif a[i, j] == 2: r[i, j] = r_end
                elif a[i, j] == 0: r[i, j] = 0

        self.grid_map = a
        self.grid_r = r

        return a, r, x_str, y_str, x_end, y_end

    def is_out(self, x, y):
        if x < 0 or x >= self.n or y < 0 or y >= self.m:
            return True
        else:
            return False

    def move(self, state, action):
        x, y = state
        if action == 0:
            x, y = x, y
        elif action == 1:
            x, y = x - 1, y
        elif action == 2:
            x, y = x + 1, y
        elif action == 3:
            x, y = x, y - 1
        elif action == 4:
            x, y = x, y + 1
        return x, y

    def pi(self, state, action, obj=None):
        if obj == None:
            obj = self.pi_tab
        if state not in obj:
            obj[state] = {}
        if action not in obj[state]:
            obj[state][action] = 0
        
        return obj[state][action]

    def state_value(self, state, obj=None):
        if obj == None:
            obj = self.state_value_tab
        if state not in obj:
            obj[state] = 0
        
        return obj[state]
    
    def action_value(self, state, action, obj=None):
        if obj == None:
            obj = self.action_value_tab
        if state not in obj:
            obj[state] = {}
        if action not in obj[state]:
            obj[state][action] = 0
        
        return obj[state][action]

    def prob_reward(self, state, action, reward):
        if state not in self.prob_reward_tab:
            self.prob_reward_tab[state] = {}
        if action not in self.prob_reward_tab[state]:
            self.prob_reward_tab[state][action] = {}
        self.prob_reward_tab[state][action][reward] = 0

        x, y = self.move(state, action)

        if  (self.is_out(x, y) and reward == self.r_out) or (not self.is_out(x, y) and reward == self.grid_r[x, y]):
            self.prob_reward_tab[state][action][reward] = 1

        return self.prob_reward_tab[state][action][reward]

    def prob_state(self, state_1, action, state_2):
        if state_1 not in self.prob_state_tab:
            self.prob_state_tab[state_1] = {}
        if action not in self.prob_state_tab[state_1]:
            self.prob_state_tab[state_1][action] = {}
        self.prob_state_tab[state_1][action][state_2] = 0

        x, y = self.move(state_1, action)
        
        if (not self.is_out(x, y) and (x, y) == state_2) or (self.is_out(x, y) and state_1 == state_2):
            self.prob_state_tab[state_1][action][state_2] = 1

        return self.prob_state_tab[state_1][action][state_2]

    def plot_start_map(self):
        fig, ax = plt.subplots()    # 创建图层、坐标轴
        ax.set_aspect('equal')      # x. y轴保持一样的比例

        # 绘制网格
        for i in range(self.n + 1):
            ax.axhline(i, color='black', linewidth=0.5)
        for i in range(self.m + 1):
            ax.axvline(i, color='black', linewidth=0.5)

        # 设置坐标轴刻度，使其位于每个格子的中间
        ax.set_xticks([j + 0.5 for j in range(self.m)])
        ax.set_yticks([i + 0.5 for i in range(self.n)])

        # 设置坐标轴刻度标签
        ax.set_xticklabels([i for i in range(self.m)])
        ax.set_yticklabels([i for i in range(self.n)])

        # 隐藏顶部和右侧的刻度线
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        # 绘制障碍物
        for i in range(self.n):
            for j in range(self.m):
                if self.grid_map[i, j] == -1:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='orange', facecolor='orange')
                    ax.add_patch(rect)

        # 绘制起点
        for i in range(self.n):
            for j in range(self.m):
                if self.grid_map[i, j] == 1:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='pink', facecolor='pink')
                    ax.add_patch(rect)

        # 绘制终点
        for i in range(self.n):
            for j in range(self.m):
                if self.grid_map[i, j] == 2:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='#ADD8E6', facecolor='#ADD8E6')
                    ax.add_patch(rect)

        # 设置轴范围
        ax.set_xlim(0, self.m)
        ax.set_ylim(0, self.n)
        ax.invert_yaxis()  # 使 (0,0) 在左上角

        plt.show()

    def upd_pi(self, state, action, val, obj=None):
        if obj == None:
            obj = self.pi_tab
        if state not in obj:
            obj[state] = {}
        if action not in obj[state]:
            obj[state][action] = 0
        obj[state][action] = val

    def upd_state_value(self, state, val, obj):
        if obj == None:
            obj = self.state_value_tab
        if state not in obj:
            obj[state] = 0
        obj[state] = val

    def upd_action_value(self, state, action, val, obj=None):
        if obj == None:
            obj = self.action_value_tab
        if state not in obj:
            obj[state] = {}
        if action not in obj[state]:
            obj[state][action] = 0
        obj[state][action] = val

    def get_state_value_Gap(self, a, b):
        sum = 0
        for state in self.s_seq:
            v1 = self.state_value(state, a)
            v2 = self.state_value(state, b)
            sum += (v1 - v2) ** 2
        return sum

    def cal_pi_state_and_action_value(self):
        # copy.deepcopy(self.b)
        # 因为字典是可变对象，所以修改pi/state_value/action_value的同时就已经在修改self.true_...了
        pi = self.true_pi_tab
        state_value = self.true_state_value_tab
        action_value = self.action_value_tab

        pi_k = copy.deepcopy(pi)
        state_value_k = copy.deepcopy(state_value)

        while True:
            # 计算action values
            for state in self.s_seq:
                for action in self.a_seq:
                    sum1 = 0
                    for r in self.r_seq:
                        sum1 += self.prob_reward(state, action, r) * r
                    sum2 = 0
                    for ss in self.s_seq:
                        sum2 += self.prob_state(state, action, ss) * self.state_value(ss, state_value)
                    sum2 *= self.gamma
                    self.upd_action_value(state, action, sum1 + sum2, action_value)
            # policy improvement
            for state in self.s_seq:
                pos = -1
                maxx = float('-inf')
                for action in self.a_seq:
                    self.upd_pi(state, action, 0, pi_k)
                    v = self.action_value(state, action, action_value)
                    if v > maxx:
                        maxx = v
                        pos = action
                self.upd_pi(state, pos, 1, pi_k)
            # state value udpate
            for state in self.s_seq:
                for action in self.a_seq:
                    if self.pi(state, action, pi_k) == 1:
                        self.upd_state_value(state, self.action_value(state, action, action_value), state_value_k)
            # 判断是否该退出
            if (self.get_state_value_Gap(state_value, state_value_k) < self.eps):
                break
            # 赋值
            pi = copy.deepcopy(pi_k)
            state_value = copy.deepcopy(state_value_k)

        self.true_pi_tab = copy.deepcopy(pi)
        self.true_state_value_tab = copy.deepcopy(state_value)
        self.true_action_value_tab = copy.deepcopy(action_value)

    def plot_standard_map(self):
        fig, ax = plt.subplots()    # 创建图层、坐标轴
        ax.set_aspect('equal')      # x. y轴保持一样的比例

        # 绘制网格
        for i in range(self.n + 1):
            ax.axhline(i, color='black', linewidth=0.5)
        for i in range(self.m + 1):
            ax.axvline(i, color='black', linewidth=0.5)

        # 设置坐标轴刻度，使其位于每个格子的中间
        ax.set_xticks([j + 0.5 for j in range(self.m)])
        ax.set_yticks([i + 0.5 for i in range(self.n)])

        # 设置坐标轴刻度标签
        ax.set_xticklabels([i for i in range(self.m)])
        ax.set_yticklabels([i for i in range(self.n)])

        # 隐藏顶部和右侧的刻度线
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        # 绘制障碍物
        for i in range(self.n):
            for j in range(self.m):
                if self.grid_map[i, j] == -1:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='orange', facecolor='orange')
                    ax.add_patch(rect)

        # 绘制起点
        for i in range(self.n):
            for j in range(self.m):
                if self.grid_map[i, j] == 1:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='pink', facecolor='pink')
                    ax.add_patch(rect)

        # 绘制终点
        for i in range(self.n):
            for j in range(self.m):
                if self.grid_map[i, j] == 2:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='#ADD8E6', facecolor='#ADD8E6')
                    ax.add_patch(rect)

        # 绘制箭头
        for i in range(self.n):
            for j in range(self.m):
                pos = -1
                for action in self.a_seq:
                    if self.pi((i,j), action, self.true_pi_tab) == 1:
                        pos = action
                if pos == 0:
                    circle = patches.Circle((j + 0.5, i + 0.5), 0.05, fc='g', ec='g')
                    ax.add_patch(circle)
                elif pos == 1:
                    ax.arrow(j + 0.5, i + 0.5, 0, -0.2, head_width=0.2, head_length=0.1, fc='g', ec='g')
                elif pos == 2:
                    ax.arrow(j + 0.5, i + 0.5, 0, 0.2, head_width=0.2, head_length=0.1, fc='g', ec='g')
                elif pos == 3:
                    ax.arrow(j + 0.5, i + 0.5, -0.2, 0, head_width=0.2, head_length=0.1, fc='g', ec='g')
                elif pos == 4:
                    ax.arrow(j + 0.5, i + 0.5, 0.2, 0, head_width=0.2, head_length=0.1, fc='g', ec='g')

        # 绘制数字
        for i in range(self.n):
            for j in range(self.m):
                number = self.state_value((i,j), self.true_state_value_tab)
                number = round(number, 1)
                ax.text(j + 0.5, i + 0.5, str(number), ha='center', va='center', color='black', fontsize=12)

        # 设置轴范围
        ax.set_xlim(0, self.m)
        ax.set_ylim(0, self.n)
        ax.invert_yaxis()  # 使 (0,0) 在左上角

        plt.show()

    def plot_end_map(self):
        # 清除当前的轴
        self.ax.clear()

        # 绘制网格
        for i in range(self.n + 1):
            self.ax.axhline(i, color='black', linewidth=0.5)
        for i in range(self.m + 1):
            self.ax.axvline(i, color='black', linewidth=0.5)

        # 设置坐标轴刻度，使其位于每个格子的中间
        self.ax.set_xticks([j + 0.5 for j in range(self.m)])
        self.ax.set_yticks([i + 0.5 for i in range(self.n)])

        # 设置坐标轴刻度标签
        self.ax.set_xticklabels([i for i in range(self.m)])
        self.ax.set_yticklabels([i for i in range(self.n)])

        # 隐藏顶部和右侧的刻度线
        self.ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        # 绘制障碍物
        for i in range(self.n):
            for j in range(self.m):
                if self.grid_map[i, j] == -1:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='orange', facecolor='orange')
                    self.ax.add_patch(rect)

        # 绘制起点
        for i in range(self.n):
            for j in range(self.m):
                if self.grid_map[i, j] == 1:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='pink', facecolor='pink')
                    self.ax.add_patch(rect)

        # 绘制终点
        for i in range(self.n):
            for j in range(self.m):
                if self.grid_map[i, j] == 2:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='#ADD8E6', facecolor='#ADD8E6')
                    self.ax.add_patch(rect)

        # 绘制箭头
        for i in range(self.n):
            for j in range(self.m):
                sum = 0
                for action in self.a_seq:
                    sum += self.pi((i,j), action)
                for action in self.a_seq:
                    ratio = self.pi((i,j), action) / sum
                    if ratio > 1e-8:
                        if action == 0:
                            circle = patches.Circle((j + 0.5, i + 0.5), 0.05 * ratio, fc='g', ec='g')
                            self.ax.add_patch(circle)
                        elif action == 1:
                            self.ax.arrow(j + 0.5, i + 0.5, 0, -0.2 * ratio, head_width=0.2, head_length=0.1, fc='g', ec='g')
                        elif action == 2:
                            self.ax.arrow(j + 0.5, i + 0.5, 0, 0.2 * ratio, head_width=0.2, head_length=0.1, fc='g', ec='g')
                        elif action == 3:
                            self.ax.arrow(j + 0.5, i + 0.5, -0.2 * ratio, 0, head_width=0.2, head_length=0.1, fc='g', ec='g')
                        elif action == 4:
                            self.ax.arrow(j + 0.5, i + 0.5, 0.2 * ratio, 0, head_width=0.2, head_length=0.1, fc='g', ec='g')

        # 绘制数字
        for i in range(self.n):
            for j in range(self.m):
                number = self.state_value((i,j))
                number = round(number, 1)
                self.ax.text(j + 0.5, i + 0.5, str(number), ha='center', va='center', color='black', fontsize=12)

        # 设置轴范围
        self.ax.set_xlim(0, self.m)
        self.ax.set_ylim(0, self.n)
        self.ax.invert_yaxis()  # 使 (0,0) 在左上角

        self.fig.canvas.draw()  # 刷新画布
        plt.pause(0.5)

    def push_state_value(self, online=False):
        self.convergence.append(self.get_state_value_Gap(self.state_value_tab, self.true_state_value_tab))
        if online == True:
            self.plot_end_map()

    def plot_end_convergence(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.convergence, linestyle='-', color='b')  # 绘制误差列表
        plt.title('Convergence Plot')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()
    
    def report(self):
        print('-------------------------------------------')
        print('迭代次数: ', len(self.convergence))
        print('-------------------------------------------')