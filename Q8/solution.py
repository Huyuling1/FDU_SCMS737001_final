import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
from collections import deque

class MazeEnvironment:
    def __init__(self, maze_file='maze_data.npz'):
        if not os.path.exists(maze_file):
            raise FileNotFoundError(f"找不到迷宫数据文件: {maze_file}")
            
        data = np.load(maze_file)
        self.grid = data['grid'] # 0=路, 1=墙
        self.start_pos = tuple(data['start'])
        self.rows, self.cols = self.grid.shape
        self.goal_pos = None
        self.current_pos = self.start_pos
        self.action_space = [0, 1, 2, 3]
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
    def is_reachable(self, start, goal):
        queue = deque([start])
        visited = set([start])
        while queue:
            current = queue.popleft()
            if current == goal: return True
            r, c = current
            for action_idx in self.action_space:
                dr, dc = self.action_map[action_idx]
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def reset_goal(self, fixed_goal=None):
        if fixed_goal:
            if not self.is_reachable(self.start_pos, fixed_goal):
                print(f"警告：终点 {fixed_goal} 不可达！")
            self.goal_pos = fixed_goal
        else:
            print("正在寻找合法的随机终点...")
            while True:
                r = random.randint(0, self.rows - 1)
                c = random.randint(0, self.cols - 1)
                if self.grid[r, c] == 0 and (r, c) != self.start_pos:
                    candidate = (r, c)
                    dist = abs(r - self.start_pos[0]) + abs(c - self.start_pos[1])
                    if 5< dist <30 and self.is_reachable(self.start_pos, candidate):
                        self.goal_pos = candidate
                        print(f"找到合法终点: {self.goal_pos} (距离: {dist})")
                        break
        return self.goal_pos

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        move = self.action_map[action]
        next_r = self.current_pos[0] + move[0]
        next_c = self.current_pos[1] + move[1]
        
        hit_wall = False
        if (next_r < 0 or next_r >= self.rows or 
            next_c < 0 or next_c >= self.cols or 
            self.grid[next_r, next_c] == 1):
            hit_wall = True
            next_state = self.current_pos 
        else:
            next_state = (next_r, next_c)
            
        self.current_pos = next_state
        
        if next_state == self.goal_pos:
            reward = 100
            done = True
        elif hit_wall:
            reward = -5 
            done = False
        else:
            reward = -1 
            done = False
        return next_state, reward, done

class QLearningSolver:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995 
        self.min_epsilon = 0.05
        
        self.best_q_table = None
        self.best_reward = -float('inf')
        
        # --- 启发式初始化 (Heuristic Initialization) ---
        # 我们不再把 Q 表初始化为平平的 -10000
        # 而是根据"离终点越近，分数越高"的原则来初始化
        # 这样 Agent 天生就知道大概往哪个方向走是对的
        self.q_table = np.zeros((env.rows, env.cols, 4))
        
        if env.goal_pos is None:
            raise ValueError("初始化 Solver 前必须先设置环境的 goal_pos")
            
        gy, gx = env.goal_pos
        
        print("正在进行启发式 Q 表初始化 (Compass)...")
        for r in range(env.rows):
            for c in range(env.cols):
                if env.grid[r, c] == 1: # 墙壁
                    self.q_table[r, c, :] = -99999 # 墙壁设为绝对不可达
                else:
                    # 计算曼哈顿距离
                    dist = abs(r - gy) + abs(c - gx)
                    # 初始分值 = -距离 * 2
                    # 离终点越近，这个负数越小（越接近0），Agent 就越喜欢去
                    heuristic_val = -dist * 2.0 
                    self.q_table[r, c, :] = heuristic_val

    def get_action(self, state, mode='train'):
        if mode == 'train' and random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            r, c = state
            values = self.q_table[r, c]
            max_val = np.max(values)
            candidates = [i for i, v in enumerate(values) if v == max_val]
            return random.choice(candidates)

    def train(self, episodes=3000, max_steps=5000):
        rewards_history = []
        print(f"开始训练 (V5 Heuristic版)...")
        print(f"起点: {self.env.start_pos} -> 终点: {self.env.goal_pos}")
        
        success_count = 0
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            episode_history = [] 

            for step in range(max_steps):
                action = self.get_action(state, mode='train')
                next_state, reward, done = self.env.step(action)
                
                episode_history.append((state, action, reward, next_state))
                
                r, c = state
                nr, nc = next_state
                old_val = self.q_table[r, c, action]
                next_max = np.max(self.q_table[nr, nc])
                new_val = old_val + self.alpha * (reward + self.gamma * next_max - old_val)
                self.q_table[r, c, action] = new_val
                
                state = next_state
                total_reward += reward
                
                if done:
                    success_count += 1
                    # V5: 依然保留反向传播，双管齐下
                    back_alpha = 0.5 
                    for (s, a, rew, s_next) in reversed(episode_history):
                        sr, sc = s
                        nr, nc = s_next
                        max_next_q = np.max(self.q_table[nr, nc])
                        current_q = self.q_table[sr, sc, a]
                        target = rew + self.gamma * max_next_q
                        self.q_table[sr, sc, a] += back_alpha * (target - current_q)
                    break
            
            # 保存最佳模型
            if done and total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_q_table = copy.deepcopy(self.q_table)
                # 只有当确实找到了很好的路径时才打印，避免刷屏
                if episode % 10 == 0 or total_reward > -500:
                    print(f"Episode {episode}: 优化路径! Reward: {total_reward} (Steps: {step+1})")

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            
            if (episode + 1) % 500 == 0:
                print(f"Ep {episode+1} | Eps: {self.epsilon:.2f} | R: {total_reward} | Total Successes: {success_count}")

        print(f"训练结束。总成功次数: {success_count}")
        
        if self.best_q_table is not None:
            print("加载最佳模型用于测试...")
            self.q_table = self.best_q_table
        
        return rewards_history

    def solve_path(self, max_steps=3000):
        state = self.env.reset()
        path = [state]
        done = False
        print("开始规划路径...")
        
        # --- V5: 简单的死锁检测 ---
        # 如果 Agent 在同一个格子停留次数过多，强制随机移动一下
        visit_counts = {}

        for _ in range(max_steps):
            # 记录访问次数
            visit_counts[state] = visit_counts.get(state, 0) + 1
            
            # 如果在同一个格子反复横跳超过 5 次，或者卡在某个死循环里
            if visit_counts[state] > 5:
                # 强制随机选一个合法的方向（破局）
                action = random.choice(self.env.action_space)
            else:
                action = self.get_action(state, mode='eval')

            next_state, _, done = self.env.step(action)
            path.append(next_state)
            state = next_state
            
            if done:
                print(f"成功到达终点！路径长度: {len(path)-1} 步")
                return path
        
        print(f"未能到达终点 (超时)，目前处于: {state}")
        return path

def visualize_solution(env, path, filename='maze_solution_v5.png'):
    plt.figure(figsize=(12, 10))
    cmap = matplotlib.colors.ListedColormap(['black', 'white'])
    plt.imshow(env.grid, cmap=cmap, origin='upper')
    
    if len(path) > 1:
        path_y, path_x = zip(*path)
        plt.plot(path_x, path_y, color='cyan', linewidth=2, alpha=0.8, label='Path')
        plt.scatter(path_x[-1], path_y[-1], c='magenta', marker='x', s=100)
    
    sy, sx = env.start_pos
    gy, gx = env.goal_pos
    plt.scatter(sx, sy, c='lime', s=150, marker='o', label='Start', edgecolors='black')
    plt.scatter(gx, gy, c='red', s=200, marker='*', label='Goal', edgecolors='black')
    
    plt.legend()
    plt.title(f"Solution V5 (Heuristic)\nBest Reward: {solver.best_reward}", fontsize=14)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"结果已保存至: {filename}")

if __name__ == "__main__":
    try:
        env = MazeEnvironment()
        env.reset_goal()
        
        solver = QLearningSolver(env, alpha=0.1, gamma=0.99)
        solver.train(episodes=5000, max_steps=5000) 
        best_path = solver.solve_path()
        visualize_solution(env, best_path)
    except Exception as e:
        print(f"发生错误: {e}")