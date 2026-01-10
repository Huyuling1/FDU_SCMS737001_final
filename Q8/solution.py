import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import os
import copy
from collections import deque

class MazeEnvironment:
    def __init__(self, maze_file='maze_data.npz'):
        if not os.path.exists(maze_file):
            print(f"提示: 未找到 {maze_file}，将生成一个临时随机迷宫用于演示。")
            self.rows, self.cols = 15, 15
            self.grid = np.zeros((self.rows, self.cols))
            self.grid[0, :] = 1
            self.grid[-1, :] = 1
            self.grid[:, 0] = 1
            self.grid[:, -1] = 1
            for _ in range(40):
                r, c = random.randint(1, self.rows-2), random.randint(1, self.cols-2)
                self.grid[r, c] = 1
            self.start_pos = (1, 1)
            self.grid[1, 1] = 0
        else:
            data = np.load(maze_file)
            self.grid = data['grid'] # 0=路, 1=墙
            self.start_pos = tuple(data['start'])
            self.rows, self.cols = self.grid.shape
            
        self.goal_pos = None
        self.current_pos = self.start_pos
        self.action_space = [0, 1, 2, 3]
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        # 死胡同
        self.dead_ends = set()
        
        # 记录已经走过的路径
        self.visited_path = set()

    def _precompute_dead_ends(self):
        """
        预计算迷宫中的所有死胡同。
        """
        if self.goal_pos is None:
            return

        temp_grid = self.grid.copy()
        sr, sc = self.start_pos
        gr, gc = self.goal_pos
        
        changed = True
        while changed:
            changed = False
            for r in range(self.rows):
                for c in range(self.cols):
                    if temp_grid[r, c] == 1 or (r, c) == (sr, sc) or (r, c) == (gr, gc):
                        continue
                    
                    walls = 0
                    for dr, dc in self.action_map.values():
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                            walls += 1
                        elif temp_grid[nr, nc] == 1:
                            walls += 1
                    
                    if walls >= 3:
                        temp_grid[r, c] = 1 
                        self.dead_ends.add((r, c))
                        changed = True
        
        print(f"共有 {len(self.dead_ends)} 个死胡同格子。")

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
                print(f"Warning: 终点 {fixed_goal} 不可达！")
            self.goal_pos = fixed_goal
        else:
            print("寻找合法的随机终点...")
            while True:
                r = random.randint(0, self.rows - 1)
                c = random.randint(0, self.cols - 1)
                if self.grid[r, c] == 0 and (r, c) != self.start_pos and self.is_reachable(self.start_pos, (r, c)):
                    self.goal_pos = (r, c)
                    dist = abs(r - self.start_pos[0]) + abs(c - self.start_pos[1])
                    print(f"找到合法终点: {self.goal_pos} (Distant: {dist})")
                    break
                else:
                    print(f"该终点无法到达: ({r}, {c})")
        
        self._precompute_dead_ends()
        return self.goal_pos

    def reset(self):
        self.current_pos = self.start_pos
        self.visited_path = set()
        self.visited_path.add(self.start_pos)
        return self.current_pos

    def step(self, action):
        move = self.action_map[action]
        next_r = self.current_pos[0] + move[0]
        next_c = self.current_pos[1] + move[1]
        
        hit_wall = False
        # 越界或撞墙检查
        if (next_r < 0 or next_r >= self.rows or 
            next_c < 0 or next_c >= self.cols or 
            self.grid[next_r, next_c] == 1):
            hit_wall = True
            next_state = self.current_pos 
        else:
            next_state = (next_r, next_c)
            
        self.current_pos = next_state
        
        is_revisit = next_state in self.visited_path
        
        if not hit_wall:
            self.visited_path.add(next_state)
        
        done = False
        
        if next_state == self.goal_pos:
            reward = 100
            done = True
        elif hit_wall:
            reward = -200
        elif next_state in self.dead_ends:
            reward = -50
        elif is_revisit:
            reward = -10 
        else:
            reward = -1
            
        return next_state, reward, done

class QLearningSolver:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.02
        
        self.best_q_table = None
        self.best_reward = -float('inf')
        
        self.q_table = np.full((env.rows, env.cols, 4), -10.0)

    def get_action(self, state, mode='train'):
        if mode == 'train' and random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            r, c = state
            values = self.q_table[r, c]
            max_val = np.max(values)
            candidates = [i for i, v in enumerate(values) if v == max_val]
            return random.choice(candidates)

    def train(self, episodes=3000, max_steps=2000):
        rewards_history = []
        print(f"开始训练...")
        print(f"起点: {self.env.start_pos} -> 终点: {self.env.goal_pos}")
        
        success_count = 0
        
        for episode in range(episodes):
                 
            state = self.env.reset()
            total_reward = 0
            done = False
            
            for step in range(max_steps):
                action = self.get_action(state, mode='train')
                next_state, reward, done = self.env.step(action)
                
                r, c = state
                nr, nc = next_state
                old_val = self.q_table[r, c, action]
                next_max = np.max(self.q_table[nr, nc])
                
                # Q-Learning 更新公式
                new_val = old_val + self.alpha * (reward + self.gamma * next_max - old_val)
                self.q_table[r, c, action] = new_val
                
                state = next_state
                total_reward += reward
                
                if done:
                    success_count += 1
                    break
            
            # 记录最佳模型
            if done and total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_q_table = copy.deepcopy(self.q_table)
                print(f"Episode {episode}: 发现新路径 Reward: {total_reward} (Steps: {step+1})")

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            
            if (episode + 1) % 500 == 0:
                print(f"Ep {episode+1} | Eps: {self.epsilon:.2f} | R: {total_reward:.2f} | Acc Success: {success_count}")

        print(f"训练结束。总成功次数: {success_count}")
        
        if self.best_q_table is not None:
            print("使用最佳模型测试...")
            self.q_table = self.best_q_table
        
        return rewards_history

    def solve_path(self, max_steps=1000):
        state = self.env.reset()
        path = [state]
        done = False
        print("开始规划路径...")
                
        for _ in range(max_steps):
            action = self.get_action(state, mode='eval')
            next_state, _, done = self.env.step(action)
            path.append(next_state)
            state = next_state
            
            if done:
                print(f"成功到达终点！路径长度: {len(path)-1} 步")
                return path
        
        print(f"未能到达终点，最终处于: {state}")
        return path

def visualize_solution(env, path, filename='maze_solution.png'):
    plt.figure(figsize=(10, 10))
    cmap = matplotlib.colors.ListedColormap(['black', 'white'])
    plt.imshow(env.grid, cmap=cmap, origin='upper')

    if len(path) > 1:
        path_y, path_x = zip(*path)
        plt.plot(path_x, path_y, color='cyan', linewidth=2, alpha=0.8, label='Agent Path')
        plt.scatter(path_x[-1], path_y[-1], c='magenta', marker='x', s=80)
    
    sy, sx = env.start_pos
    if env.goal_pos:
        gy, gx = env.goal_pos
        plt.scatter(gx, gy, c='red', s=150, marker='*', label='Goal', edgecolors='white')
        
    plt.scatter(sx, sy, c='lime', s=100, marker='o', label='Start', edgecolors='black')
    
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"路径图保存至: {filename}")

if __name__ == "__main__":
    try:
        env = MazeEnvironment()
        
        env.reset_goal() 
        
        solver = QLearningSolver(env, alpha=0.1, gamma=0.99)
        
        # 训练
        solver.train(episodes=2000, max_steps=1000) 
        
        # 测试
        best_path = solver.solve_path()
        
        # 可视化
        visualize_solution(env, best_path)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"发生错误: {e}")