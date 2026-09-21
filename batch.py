import cv2
import numpy as np
import os
import glob
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import itertools
import time

# ==========================================
# 1. 核心运行与文件路径配置及参数定义
# ==========================================
DATASET_DIR = "./capture_and_calib/SGM_dataset/bottle6.22"  # 数据集根目录
LEFT_IMG_DIR = os.path.join(DATASET_DIR, "cam0")
RIGHT_IMG_DIR = os.path.join(DATASET_DIR, "cam1")
VIO_CSV_PATH = os.path.join(DATASET_DIR, "vio.csv")
OUTPUT_LOCAL_DIR = os.path.join(DATASET_DIR, "local_pcds")

# ----------------- 【核心控制参数】 -----------------
VIO_INTERVAL = 1

# ----------------- 【自动化网格搜索(Grid Search)参数池】 -----------------
SGM_BLOCK_SIZE = 7

# ----------------- 【学术级 SGBM 消融实验配置矩阵】 -----------------
# 格式: (Multiplier, Uniqueness, MaxDiff, "实验命名标签")
EXPERIMENT_CONFIGS = [
    # --- 1. 对照组 (Baseline) ---
    (2, 10, 10, "Baseline_Normal"),    # 正常约束：各方面适中，作为参考系

    # --- 2. 控制变量：平滑惩罚倍数 (Multiplier) ---
    (5, 10, 10, "Loose_Multiplier"),   # 极度宽松：强行平滑弱纹理 (观察：拉丝/肥大)
    (1, 10, 10, "Strict_Multiplier"),  # 极度严苛：拒绝平滑扩散 (观察：圆柱体破洞)

    # --- 3. 控制变量：置信度门槛 (Uniqueness) ---
    (2,  5, 10, "Loose_Uniqueness"),   # 极度宽松：允许算法瞎猜 (观察：空洞减少，但噪点增加)
    (2, 15, 10, "Strict_Uniqueness"),  # 极度严苛：只信满分匹配 (观察：极度纯净，但点云稀疏)

    # --- 4. 控制变量：左右一致性校验 (MaxDiff) ---
    (2, 10, 32, "Loose_MaxDiff"),      # 极度宽松：无视左右眼差异 (观察：边缘产生长条飞线)
    (2, 10,  2, "Strict_MaxDiff"),     # 极度严苛：左右眼必须完全一致 (观察：边缘闭塞区产生黑洞)

    # --- 5. 终极叠加态 (用于展示极端系统行为) ---
    (5,  5, 32, "Ultimate_Loose"),     # 终极弱纹理狂暴模式：底线全抛 (观察：最密集的点云，但也最容易和背景糊成一团)
    (1, 15,  2, "Ultimate_Strict")     # 终极保守纯净模式：绝不容错 (观察：极其干净的少量碎块点云，完全无法拼成物体)
]

# ----------------- 【物理空间裁剪参数】 -----------------
MIN_Z_DIST = 300.0
MAX_Z_DIST = 400.0

# ----------------- 【体素下采样参数】 (防爆内存必须保留) -----------------
VOXEL_SIZE_LOCAL = 0.05
VOXEL_SIZE_GLOBAL = 0.1

# ==========================================
# 2. 硬件标定硬编码 (2026-06-03 最新标定)
# ==========================================
M1 = np.array([[1289.72072592, 0.0, 658.06396116], [0.0, 1291.10293811, 512.78197975], [0.0, 0.0, 1.0]])
d1 = np.array([-0.06674887, 0.09300343, 0.00510734, 0.00472166])
M2 = np.array([[1300.44973019, 0.0, 607.63375397], [0.0, 1299.81474378, 488.45282347], [0.0, 0.0, 1.0]])
d2 = np.array([-0.05535563, 0.05703770, -0.00148756, -0.00268021])

T_cam0_to_cam1_meters = np.array([
    [0.76598923, -0.00812042, 0.64280212, -0.15139154],
    [-0.00050253, 0.99991235, 0.01323057, 0.00279326],
    [-0.64285321, -0.01045750, 0.76591800, 0.05435582],
    [0.0, 0.0, 0.0, 1.0]
])
T_cam0_to_cam1_mm = T_cam0_to_cam1_meters.copy()
T_cam0_to_cam1_mm[:3, 3] *= 1000.0

T_cam0_to_imu_meters = np.array([
    [-0.93940282, 0.00190882, -0.34281001, 0.083887],
    [0.34275999, 0.02319650, -0.93913658, -0.06495405],
    [0.00615935, -0.99972910, -0.02244512, 0.00076514],
    [0.0, 0.0, 0.0, 1.0]
])
T_cam0_to_imu_mm = T_cam0_to_imu_meters.copy()
T_cam0_to_imu_mm[:3, 3] *= 1000.0


# ==========================================
# 3. 基础函数与 SGM 算法模块
# ==========================================
def get_rectification_maps(size):
    R_mat = T_cam0_to_cam1_mm[:3, :3]
    t_vec = T_cam0_to_cam1_mm[:3, 3].reshape(3, 1)
    R1, R2, P1, P2, Q = cv2.stereoRectify(
        cameraMatrix1=M1, cameraMatrix2=M2, distCoeffs1=d1, distCoeffs2=d2,
        R=R_mat, T=t_vec, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1, imageSize=size, newImageSize=size
    )[0:5]
    map1x, map1y = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_32FC1)
    return {'map1x': map1x, 'map1y': map1y, 'map2x': map2x, 'map2y': map2y, 'Q': Q, 'R1': R1}


def load_vins_trajectory(csv_path):
    print(f"[INFO] 正在加载 VINS 轨迹: {csv_path}")
    df = pd.read_csv(csv_path, header=None, usecols=range(8))
    df.columns = ['timestamp', 'p_x', 'p_y', 'p_z', 'q_w', 'q_x', 'q_y', 'q_z']
    if df['timestamp'].iloc[0] > 1e12:
        df['timestamp'] = df['timestamp'] / 1e9
    return df


def global_estimate_sgm_params(valid_images, right_dir, rect_maps):
    print(f"\n[INFO] === 开始进行全局视差边界预估 (该步骤只需运行一次) ===")
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    all_disparities = []
    sample_step = max(1, len(valid_images) // 100)

    for idx in range(0, len(valid_images), sample_step):
        img_name = os.path.basename(valid_images[idx])
        imgR_path = os.path.join(right_dir, img_name)
        if not os.path.exists(imgR_path): continue

        imgL = cv2.imread(valid_images[idx], 0)
        imgR = cv2.imread(imgR_path, 0)
        rectL = cv2.remap(imgL, rect_maps['map1x'], rect_maps['map1y'], cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, rect_maps['map2x'], rect_maps['map2y'], cv2.INTER_LINEAR)

        kp1, des1 = orb.detectAndCompute(rectL, None)
        kp2, des2 = orb.detectAndCompute(rectR, None)
        if des1 is None or des2 is None: continue

        matches = bf.match(des1, des2)
        disparities = [kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]
                       for m in matches if abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) < 2.0]
        all_disparities.extend(disparities)

    if not all_disparities:
        return 0, 256

    disparities_arr = np.array(all_disparities)
    min_val = np.percentile(disparities_arr, 1)
    max_val = np.percentile(disparities_arr, 99)
    padding = 32.0

    global_min_disp = int(np.floor((min_val - padding) / 16.0) * 16)
    global_num_disp = int(np.ceil((max_val + padding - global_min_disp) / 16.0) * 16)
    global_min_disp = max(-32, global_min_disp)
    global_num_disp = max(128, global_num_disp)

    print(f"[INFO] 全局预估完成！ minDisp={global_min_disp}, numDisp={global_num_disp}\n")
    return global_min_disp, global_num_disp


def compute_sgm_disparity(rectL, rectR, min_disp, num_disp, block_size, multiplier, max_diff, uniqueness):
    # 【修复】：函数内部必须根据传入的 multiplier 动态计算 P1 和 P2
    P1 = 4 * multiplier * block_size ** 2
    P2 = 16 * multiplier * block_size ** 2

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size,
        P1=P1, P2=P2,
        disp12MaxDiff=max_diff, uniquenessRatio=uniqueness,
        speckleWindowSize=200, speckleRange=2, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    disparity_left = left_matcher.compute(rectL, rectR)
    return disparity_left.astype(np.float32) / 16.0


def generate_local_pcd(disparity, gray_img, Q):
    points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True).reshape(-1, 3)
    disp_flat = disparity.reshape(-1)
    gray_flat = gray_img.reshape(-1)

    # 关闭了梯度剔除掩码，呈现最纯粹 Raw 点云
    mask = (
            (np.isfinite(points_3d[:, 0])) &
            (disp_flat > 0) &
            (points_3d[:, 2] > MIN_Z_DIST) &
            (points_3d[:, 2] < MAX_Z_DIST) &
            (gray_flat > 3) &
            (gray_flat < 255)
    )

    valid_points = points_3d[mask]
    if len(valid_points) == 0: return o3d.geometry.PointCloud()

    valid_gray = gray_flat[mask].astype(np.float64) / 255.0
    colors_rgb = np.stack((valid_gray, valid_gray, valid_gray), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    return pcd


# ==========================================
# 4. 主干流水线 (自动化网格搜索)
# ==========================================
def main():
    os.makedirs(OUTPUT_LOCAL_DIR, exist_ok=True)
    df_traj = load_vins_trajectory(VIO_CSV_PATH)

    print("[INFO] 正在解析图像序列时间戳...")
    left_images_raw = sorted(
        glob.glob(os.path.join(LEFT_IMG_DIR, "*.png")) + glob.glob(os.path.join(LEFT_IMG_DIR, "*.bmp")))
    if not left_images_raw:
        print("[ERROR] 没有找到图像文件！")
        return

    valid_img_paths = []
    img_timestamps = []
    for img_path in left_images_raw:
        try:
            ts = int(os.path.basename(img_path).split('.')[0]) / 1e9
            img_timestamps.append(ts)
            valid_img_paths.append(img_path)
        except ValueError:
            continue
    img_timestamps = np.array(img_timestamps)

    # ----- 核心加速：所有耗时的初始化工作，全部在循环外部做完！ -----
    sample_img = cv2.imread(valid_img_paths[0], 0)
    rect_maps = get_rectification_maps((sample_img.shape[1], sample_img.shape[0]))
    Q, R1 = rect_maps['Q'], rect_maps['R1']

    GLOBAL_MIN_DISP, GLOBAL_NUM_DISP = global_estimate_sgm_params(valid_img_paths, RIGHT_IMG_DIR, rect_maps)

    # 提前计算所有排列组合的总数，用于打印进度条
    total_combinations = len(EXPERIMENT_CONFIGS)
    current_combo_idx = 0
    start_time_total = time.time()

    # ================== 【开始精准定向消融实验】 ==================
    for mult, uniq, diff, exp_label in EXPERIMENT_CONFIGS:
        current_combo_idx += 1

        # 动态计算真实 P1 和 P2 写入文件名，并加上实验标签！
        actual_p1 = 4 * mult * SGM_BLOCK_SIZE ** 2
        actual_p2 = 16 * mult * SGM_BLOCK_SIZE ** 2
        generated_filename = f"EXP{current_combo_idx}_{exp_label}_P1-{actual_p1}_P2-{actual_p2}_uniq{uniq}_diff{diff}.ply"
        OUTPUT_GLOBAL_PLY = os.path.join(DATASET_DIR, generated_filename)

        print(f"\n=======================================================================")
        print(f"[Ablation Study {current_combo_idx}/{total_combinations}] 当前实验: 【{exp_label}】")
        print(f" -> 参数: Multiplier={mult}, Uniqueness={uniq}, MaxDiff={diff}")
        print(f" -> 输出: {generated_filename}")
        print(f"=======================================================================")


        # 每轮实验都必须重新初始化全局点云
        global_pcd = o3d.geometry.PointCloud()
        processed_count = 0

        # 遍历轨迹进行处理
        for idx in range(0, len(df_traj), VIO_INTERVAL):
            row = df_traj.iloc[idx]
            pose_ts = row['timestamp']

            time_diffs = np.abs(img_timestamps - pose_ts)
            closest_img_idx = np.argmin(time_diffs)
            if time_diffs[closest_img_idx] > 0.03: continue

            imgL_path = valid_img_paths[closest_img_idx]
            img_name = os.path.basename(imgL_path)
            imgR_path = os.path.join(RIGHT_IMG_DIR, img_name)
            if not os.path.exists(imgR_path): continue

            # 读取图片并重映射
            imgL = cv2.imread(imgL_path, 0)
            imgR = cv2.imread(imgR_path, 0)
            rectL = cv2.remap(imgL, rect_maps['map1x'], rect_maps['map1y'], cv2.INTER_LINEAR)
            rectR = cv2.remap(imgR, rect_maps['map2x'], rect_maps['map2y'], cv2.INTER_LINEAR)

            # 计算视差并生成局部点云
            disp = compute_sgm_disparity(rectL, rectR, GLOBAL_MIN_DISP, GLOBAL_NUM_DISP, SGM_BLOCK_SIZE, mult,
                                         diff, uniq)
            local_pcd = generate_local_pcd(disp, rectL, Q)
            if len(local_pcd.points) == 0: continue

            # 仅下采样，关闭单帧滤波
            local_pcd = local_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_LOCAL)

            # 位姿转换
            T_rect0_to_raw0 = np.eye(4)
            T_rect0_to_raw0[:3, :3] = R1.T
            T_wi = np.eye(4)
            r = R.from_quat([row['q_x'], row['q_y'], row['q_z'], row['q_w']])
            T_wi[:3, :3] = r.as_matrix()
            T_wi[:3, 3] = [row['p_x'] * 1000.0, row['p_y'] * 1000.0, row['p_z'] * 1000.0]

            T_wi_ic = np.dot(T_wi, T_cam0_to_imu_mm)
            T_final = np.dot(T_wi_ic, T_rect0_to_raw0)
            local_pcd.transform(T_final)

            global_pcd += local_pcd
            processed_count += 1

            if processed_count % 100 == 0:
                print(f"    ... 已融合 {processed_count} 帧 ...")

        # 单轮参数实验结束，保存全局点云
        if processed_count > 0:
            global_pcd = global_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_GLOBAL)

            T_align = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            global_pcd.transform(T_align)

            o3d.io.write_point_cloud(OUTPUT_GLOBAL_PLY, global_pcd)
            print(f"✅ [实验 {current_combo_idx} 成功] 模型已落盘，融合帧数: {processed_count}")

            # ⚠️ 极其重要：已将 draw_geometries 可视化注释掉！
            # 否则会阻塞自动化脚本的运行，导致你一晚上的网格搜索卡在第一个模型弹窗上。
            # o3d.visualization.draw_geometries([global_pcd])

    print(
        f"\n🎉 [全自动网格搜索完成] 共运行 {total_combinations} 组参数组合，总耗时 {int(time.time() - start_time_total)} 秒。")
    print(f"请打开 {DATASET_DIR} 目录，将其中的所有 .ply 拖入 CloudCompare 进行论文学术对比！")


if __name__ == "__main__":
    main()