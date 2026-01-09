import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_maze_by_darkness(image_path, grid_rows=49, grid_cols=65):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return None, None

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算步长
    dy = h / grid_rows
    dx = w / grid_cols
    
    # 初始化矩阵 (默认全是墙=1)
    maze_matrix = np.ones((grid_rows, grid_cols), dtype=int)
    
    # 2. 遍历每一个格子
    for r in range(grid_rows):
        for c in range(grid_cols):
            # 计算坐标
            y_start = int(r * dy)
            y_end = int((r + 1) * dy)
            x_start = int(c * dx)
            x_end = int((c + 1) * dx)
            
            # --- 关键改进：只提取格子中心区域 (Center Crop) ---
            # 这样可以避开网格边缘的红线或相邻格子的干扰
            # 每个边缘切掉 2 像素
            margin = 2 
            if (y_end - y_start) > 2 * margin and (x_end - x_start) > 2 * margin:
                roi = gray[y_start+margin : y_end-margin, x_start+margin : x_end-margin]
            else:
                roi = gray[y_start:y_end, x_start:x_end]
            
            # --- 核心逻辑：判断是否为“黑色” ---
            # 计算区域内的平均亮度
            # 纯黑的路，平均亮度通常 < 10
            # 带有白色条纹的墙，平均亮度通常 > 80
            mean_brightness = np.mean(roi)
            
            # 设定阈值为 45 (在 0-255 之间)
            # 只要够黑，就是路
            if mean_brightness < 45:
                maze_matrix[r, c] = 0  # 0 = Path (黑色)
            else:
                maze_matrix[r, c] = 1  # 1 = Wall (白色)

    # 3. 再次定位起点 (黄色笑脸) - 保持原有逻辑
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([45, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    points = cv2.findNonZero(mask_yellow)
    
    start_pos = None
    if points is not None:
        avg_x = np.mean(points[:, 0, 0])
        avg_y = np.mean(points[:, 0, 1])
        sr = int(avg_y / dy)
        sc = int(avg_x / dx)
        sr = min(sr, grid_rows - 1)
        sc = min(sc, grid_cols - 1)
        
        # 强制把起点设为路
        maze_matrix[sr, sc] = 0
        start_pos = (sr, sc)
    else:
        # 备用起点
        start_pos = (grid_rows-2, grid_cols-2)
        maze_matrix[start_pos[0], start_pos[1]] = 0

    return maze_matrix, start_pos

def visualize_final_check(matrix, start_pos):
    plt.figure(figsize=(12, 10))
    # 0=黑(路), 1=白(墙)
    plt.imshow(matrix, cmap='gray', origin='upper')
    
    if start_pos:
        plt.scatter(start_pos[1], start_pos[0], c='red', s=80, marker='s')
        plt.text(start_pos[1], start_pos[0], 'S', color='white', ha='center', va='center', fontweight='bold')
        
    save_name = 'maze_check.png'
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"提取完成！请查看: {save_name}")

# --- 执行 ---
maze_grid, start_pos = extract_maze_by_darkness('maze.jpg', grid_rows=49, grid_cols=65)

print(f"起点坐标: {start_pos}")
visualize_final_check(maze_grid, start_pos)

np.savez('maze_data.npz', grid=maze_grid, start=start_pos)