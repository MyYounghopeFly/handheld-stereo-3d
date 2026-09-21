import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d
import pyvista as pv


def main():
    # 1. 初始化文件选择对话框
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # 确保对话框弹出在最上层

    file_path = filedialog.askopenfilename(
        title="选择点云文件",
        filetypes=[("点云文件", "*.pcd *.ply *.xyz *.pts")]
    )

    if not file_path:
        print("未选择文件，程序退出。")
        return

    print(f"正在加载点云: {file_path}")

    # 2. 用 Open3D 读取
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        print("点云加载失败或为空！")
        return

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # 3. 转换为 PyVista 渲染结构
    cloud = pv.PolyData(points)
    if len(colors) > 0:
        cloud['RGB'] = colors

    # 4. 初始化 3D 交互窗口
    plotter = pv.Plotter(title="点云物理距离高精度测算工具")

    if len(colors) > 0:
        plotter.add_mesh(cloud, scalars='RGB', rgb=True, point_size=2, render_points_as_spheres=True)
    else:
        plotter.add_mesh(cloud, color='lightblue', point_size=2, render_points_as_spheres=True)

    # 状态缓存变量
    picked_points = []
    point_actors = []
    line_actor = None
    label_actor = None

    # 5. 【已修复】核心逻辑：定义点选回调函数
    # PyVista 现在的 enable_point_picking 会直接返回拾取到的点坐标 [x, y, z]
    def on_point_picked(point):
        nonlocal line_actor, label_actor

        # 将返回的 point 转为 numpy 数组以便进行数学计算
        pt = np.array(point)

        # 如果已经测完一次（选了两个点），再次点击时清空之前的标识和连线
        if len(picked_points) >= 2:
            picked_points.clear()
            for actor in point_actors:
                plotter.remove_actor(actor)
            point_actors.clear()
            if line_actor:
                plotter.remove_actor(line_actor)
            if label_actor:
                plotter.remove_actor(label_actor)

        picked_points.append(pt)

        # 添加高亮标识 (2D 屏幕空间自适应)
        marker = pv.PolyData(pt)
        actor = plotter.add_mesh(marker, color='red', point_size=15, render_points_as_spheres=True)
        point_actors.append(actor)

        # 当选中两个点时，触发计算和显示
        if len(picked_points) == 2:
            p1, p2 = picked_points
            dist = np.linalg.norm(p1 - p2)

            print("-" * 30)
            print(f"起点坐标: {p1}")
            print(f"终点坐标: {p2}")
            print(f"--> 真实物理距离: {dist:.4f}")

            # 绘制两点之间的 3D 黄色连线
            line = pv.Line(p1, p2)
            line_actor = plotter.add_mesh(line, color='yellow', line_width=3)

            # 自适应文本标签显示
            mid_point = (p1 + p2) / 2.0
            label_actor = plotter.add_point_labels(
                [mid_point], [f"Dist: {dist:.3f}"],
                point_size=0, font_size=20, text_color='white',
                shape_color='black', shape_opacity=0.7,
                always_visible=True
            )

    # 6. 绑定交互事件
    plotter.enable_point_picking(
        callback=on_point_picked,
        show_message="操作说明：\n1. 旋转: 左键拖动 | 平移: Shift+左键 | 缩放: 滚轮\n2. 测距: 将鼠标悬停在特征点上，按键盘 'P' 键进行精确拾取。",
        font_size=12,
        color='white'
    )

    plotter.show()


if __name__ == "__main__":
    main()