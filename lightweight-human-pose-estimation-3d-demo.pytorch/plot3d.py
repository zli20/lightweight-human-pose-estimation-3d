import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# 读取TXT文件中的点数据
def load_points_from_txt(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.strip().split())
            points.append((x, y, z))
    return np.array(points)

SKELETON_EDGES = np.array([[11, 10], [10, 9], [9, 0], [0, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [0, 12],
                           [12, 13], [13, 14], [0, 1], [1, 15], [15, 16], [1, 17], [17, 18]])

# 示例TXT文件路径
# file_path = r'E:\svn_suanfa\MotionCapture\pose3d.txt'
file_path = 'out_3dpoints.txt'
# 加载点数据
points = load_points_from_txt(file_path)

# 计算n的值
n = points.shape[0] // 19

# reshape数据
reshaped_points = points.reshape(n, 19, 3)

# 创建一个3D绘图对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # 设置轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
ax.axis('off')
ax.view_init(elev=30, azim=-45)  # 设置视角
# 绘制点和连线
for segment in reshaped_points:
    x_vals = -segment[:, 2]
    y_vals = segment[:, 0]
    z_vals = -segment[:, 1]

    # 绘制点
    ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')

    for edge in SKELETON_EDGES:
        xs = [-segment[edge[0]][2], -segment[edge[1]][2]]
        ys = [segment[edge[0]][0], segment[edge[1]][0]]
        zs = [-segment[edge[0]][1], -segment[edge[1]][1]]
        ax.plot(xs, ys, zs, c='b')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(elev=30, azim=45)
    ax.axis('off')
    plt.draw()
    plt.pause(0.01)

    # 清空绘图区
    ax.cla()

# 显示图形
plt.show()
