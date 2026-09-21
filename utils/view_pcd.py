import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os


def main():
    root = tk.Tk()
    root.withdraw()

    print("请在弹出的窗口中选择点云文件...")
    file_path = filedialog.askopenfilename(
        title="请选择要查看的 3D 点云文件",
        filetypes=[
            ("Point Cloud Files", "*.ply *.pcd"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        print("操作已取消。")
        return

    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        print("[ERROR] 点云读取失败！")
        return

    # ================= 1. 核心计算：获取轴向包围盒 (AABB) =================
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)  # 将包围盒的线条设为红色，醒目显示

    # 获取坐标边界（相对于光心 0,0,0 的绝对坐标）
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    extent = aabb.get_extent()  # 包围盒的长宽高（模型自身的三维尺寸）
    center = aabb.get_center()

    # ================= 2. 打印空间位置报告 =================
    print("\n================= 空间位置与尺寸分析 (单位: mm) =================")
    print("光心 (相机物理位置) 位于坐标原点: (0.00, 0.00, 0.00)\n")

    print(f"【X轴 (左右距离)】: {min_bound[0]:.2f} mm 到 {max_bound[0]:.2f} mm")
    print(f"  -> 模型实际宽度: {extent[0]:.2f} mm\n")

    print(f"【Y轴 (上下距离)】: {min_bound[1]:.2f} mm 到 {max_bound[1]:.2f} mm")
    print(f"  -> 模型实际高度: {extent[1]:.2f} mm\n")

    print(f"【Z轴 (纵深距离)】: {min_bound[2]:.2f} mm 到 {max_bound[2]:.2f} mm")
    print(f"  -> (这是最核心的数据：代表物体距离相机镜头有 {min_bound[2]:.2f} 毫米)")
    print(f"  -> 模型实际厚度: {extent[2]:.2f} mm\n")

    print(f"几何中心绝对坐标: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print("=================================================================")

    # ================= 3. 创建坐标系参照物 =================
    # 把坐标轴建大一点（300mm），方便跨越空间观察
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300.0, origin=[0, 0, 0])

    print("\n>>> 可视化窗口已打开。")
    print(">>> 你将看到：")
    print("    1. 原点交汇的红绿蓝三轴 (光心位置)")
    print("    2. 红色的立方体框 (点云空间边界盒)")

    # 提取文件名作为窗口标题
    window_title = f"Spatial Viewer - {os.path.basename(file_path)}"

    # 将 点云、包围盒、坐标轴 一起渲染
    o3d.visualization.draw_geometries(
        [pcd, aabb, mesh_frame],
        window_name=window_title,
        width=1280,
        height=720
    )


if __name__ == "__main__":
    main()