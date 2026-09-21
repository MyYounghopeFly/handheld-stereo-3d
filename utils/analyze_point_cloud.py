import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os


def select_point_cloud_file():
    """
    弹出一个文件选择对话框，让用户选择一个点云文件 (txt, ply, etc.)。
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="请选择一个点云文件",
        # 【修改点】: 增加了对PLY格式的支持
        filetypes=[
            ("Point Cloud Files", "*.ply *.txt *.xyz *.csv"),
            ("PLY Files", "*.ply"),
            ("Text Files", "*.txt"),
            ("XYZ Files", "*.xyz"),
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
    )
    return file_path


def analyze_point_cloud_resolution(file_path):
    """
    加载点云文件(TXT或PLY)，移除重复点后，计算其分辨率、均匀性及最小点间距。
    """
    print(f"--- 开始处理文件: {file_path} ---")

    # --- 【核心修改】: 根据文件扩展名选择加载方式 ---
    try:
        # 1. 获取文件扩展名
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension in ['.txt', '.xyz', '.csv']:
            # --- 分支1: 处理TXT等文本文件 ---
            print(f"1. 检测到文本文件格式 ({file_extension})，正在使用Numpy加载...")
            data = np.loadtxt(file_path)

            if data.ndim == 1:
                if data.shape[0] < 3:
                    raise ValueError("数据列数少于3。")
                data = data.reshape(1, -1)
            elif data.shape[1] < 3:
                raise ValueError("数据列数少于3。")

            points = data[:, :3]
            print(f"   成功加载数据，共 {len(points)} 个点。已提取前3列作为坐标。")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

        elif file_extension == '.ply':
            # --- 分支2: 处理PLY文件 ---
            print(f"1. 检测到PLY文件格式，正在使用Open3D加载...")
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                raise ValueError("PLY文件为空或不包含点数据。")
            print(f"   成功加载PLY文件，共 {len(pcd.points)} 个点。")

        else:
            # --- 分支3: 不支持的格式 ---
            raise TypeError(f"不支持的文件格式: '{file_extension}'。请选择 .txt, .xyz, .csv 或 .ply 文件。")

    except Exception as e:
        print(f"错误：加载或解析文件失败: {e}")
        return None
    # --- 【核心修改结束】 ---

    # 2. 【新增步骤】移除重复的点
    points_before_deduplication = len(pcd.points)
    print(f"2. 正在移除重复点...")
    pcd.remove_duplicated_points()
    points_after_deduplication = len(pcd.points)
    removed_count = points_before_deduplication - points_after_deduplication
    print(f"   移除了 {removed_count} 个重复点。剩余唯一有效点: {points_after_deduplication} 个。")

    if points_after_deduplication < 2:
        print("错误：去重后剩余点数不足2个，无法计算点间距。")
        return None

    # 3. 在去重后的点云上计算最近邻距离
    print("3. 正在计算最近邻距离...")
    distances = pcd.compute_nearest_neighbor_distance()

    # 4. 计算各项指标
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    min_distance = np.min(distances)

    print("--- 处理完成 ---")

    # 5. 返回结果
    return {
        "mean_distance": mean_distance,
        "std_deviation": std_distance,
        "min_distance": min_distance,
        "point_cloud": pcd
    }


if __name__ == "__main__":
    selected_file = select_point_cloud_file()
    if selected_file:
        print(f"已选择文件: {selected_file}")
        results = analyze_point_cloud_resolution(selected_file)
        if results:
            print("\n--- 点云分析结果 ---")
            print(f"分辨率 (平均最近邻距离): {results['mean_distance']:.6f}")
            print(f"均匀性 (距离的标准差):  {results['std_deviation']:.6f}")
            print(f"最小点间距 (非重叠点):   {results['min_distance']:.6f}")
            print("\n--- 指标说明 ---")
            print("  - '分辨率'：值越小，代表点云整体越密集。")
            print("  - '均匀性'：值越小，代表点云的疏密分布越均匀。")
            print("  - '最小点间距'：表示整个数据集中最靠近的两个【不同位置】的点之间的距离。")

            print("\n正在打开可视化窗口（显示的是去重后的点云），按 'q' 键关闭...")
            o3d.visualization.draw_geometries(
                [results['point_cloud']],
                window_name=f"Point Cloud from {os.path.basename(selected_file)} (Deduplicated)"
            )
    else:
        print("未选择任何文件，程序已退出。")