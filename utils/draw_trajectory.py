import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog


# ==========================================
# 1. 核心解析逻辑 (参考 sgm_stitcher.py)
# ==========================================
def load_vins_trajectory(csv_path):
    print(f"[INFO] 正在加载 VINS 轨迹: {csv_path}")
    # 使用 usecols=range(8) 防止读取到多余的列导致报错
    df = pd.read_csv(csv_path, header=None, usecols=range(8))
    df.columns = ['timestamp', 'p_x', 'p_y', 'p_z', 'q_w', 'q_x', 'q_y', 'q_z']

    # 自动处理纳秒到秒的时间戳转换
    if df['timestamp'].iloc[0] > 1e12:
        print("[INFO] 检测到轨迹时间戳为纳秒，正在自动转换为秒...")
        df['timestamp'] = df['timestamp'] / 1e9

    return df


# ==========================================
# 2. 3D 可视化与误差分析
# ==========================================
def plot_vio_trajectory(csv_path):
    try:
        df = load_vins_trajectory(csv_path)
    except Exception as e:
        print(f"[错误] 读取文件失败，请确认格式。详细信息: {e}")
        return

    # 提取空间坐标 (以米为单位)
    x, y, z = df['p_x'].values, df['p_y'].values, df['p_z'].values

    # 创建 3D 画布
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制完整轨迹线
    ax.plot(x, y, z, label='VIO Trajectory', color='b', linewidth=2)

    # 重点标记起点和终点
    ax.scatter(x[0], y[0], z[0], c='g', marker='o', s=100, label='Start Point')
    ax.scatter(x[-1], y[-1], z[-1], c='r', marker='x', s=100, label='End Point')

    # 核心指标：计算起点到终点的绝对物理误差（闭环漂移）
    drift_error_meters = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2 + (z[-1] - z[0]) ** 2)

    ax.set_title(f"VINS-Fusion 3D Trajectory\nEnd-to-End Drift: {drift_error_meters * 100:.2f} cm")
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()

    # 强制让三个轴的显示比例一致，防止视觉拉伸导致轨迹变形
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


# ==========================================
# 3. 交互式文件选择器
# ==========================================
if __name__ == "__main__":
    # 初始化 tkinter，并隐藏毫无用处的主窗口
    root = tk.Tk()
    root.withdraw()

    print("等待选择文件...")
    # 弹出文件选择对话框
    selected_file_path = filedialog.askopenfilename(
        title="请选择 VINS 输出的轨迹文件 (vio.csv)",
        filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
    )

    # 如果用户没有点击取消，则执行绘图
    if selected_file_path:
        plot_vio_trajectory(selected_file_path)
    else:
        print("未选择任何文件，程序退出。")