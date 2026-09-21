import cv2
import numpy as np
import os
import glob
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 核心运行与文件路径配置及参数定义
# ==========================================
DATASET_DIR = "./capture_and_calib/SGM_dataset/bottle9.14"  # 数据集根目录
LEFT_IMG_DIR = os.path.join(DATASET_DIR, "cam0")
RIGHT_IMG_DIR = os.path.join(DATASET_DIR, "cam1")
VIO_CSV_PATH = os.path.join(DATASET_DIR, "vio.csv")
BASE_LOCAL_DIR = os.path.join(DATASET_DIR, "local_pcds")

# ----------------- 【核心控制参数】 -----------------
# 设为 1 代表：把 vio.csv 里的每一帧有效位姿全部转化为点云糊上去（极限稠密度）
VIO_INTERVAL = 2

# ----------------- 【SGBM 实验核心参数 (用于控制变量实验)】 -----------------
SGM_BLOCK_SIZE = 25
SGM_MULTIPLIER = 1          # 决定 P1 和 P2 的平滑惩罚倍数 (建议测试: 1, 2, 5, 8)
SGM_UNIQUENESS = 25         # 置信度/唯一性比率 (建议测试: 3, 5, 10, 15)
SGM_MAX_DIFF = 1            # 左右一致性校验容差 (建议测试: 2, 5, 32, 64)

# ================== 【手动视差范围参数】 ==================
MANUAL_MIN_DISP = 486      # 最小视差，对应最远工作距离
MANUAL_NUM_DISP = 128       # 视差搜索范围，值越大能看清越近的物体

# 算出真实的 P1 和 P2 数值，以便写入文件名
ACTUAL_P1 = 8 * SGM_MULTIPLIER * SGM_BLOCK_SIZE ** 2
ACTUAL_P2 = 32 * SGM_MULTIPLIER * SGM_BLOCK_SIZE ** 2

# ----------------- 【物理空间裁剪参数】 -----------------
MIN_Z_DIST = 100.0  # 最小有效距离 (mm)
MAX_Z_DIST = 500.0  # 最大有效距离 (mm)

# ----------------- 【体素下采样参数】 -----------------
VOXEL_SIZE = 0.5  # 点云下采样体素 (mm)

# ----------------- 【去噪滤波参数】 -----------------
ROR_NB_POINTS = 100  # 半径内至少需要的邻居点数
ROR_RADIUS = 10.0  # 搜索半径 (mm)

SOR_NB_NEIGHBORS = 30  # 考察的邻居点数
SOR_STD_RATIO = 1.0  # 剔除标准差的阈值乘数

# ================== 【动态生成全局输出文件名】 ==================
generated_foldername = f"global_blk{SGM_BLOCK_SIZE}_mult{SGM_MULTIPLIER}_uniq{SGM_UNIQUENESS}_diff{SGM_MAX_DIFF}_vox{VOXEL_SIZE}"
OUTPUT_LOCAL_DIR = os.path.join(BASE_LOCAL_DIR, generated_foldername)
print(f"[INIT] 动态配置加载完毕...")
print(f"       -> 局部点云将被统一存放在独立文件夹: {OUTPUT_LOCAL_DIR}")


# ==========================================
# 硬件标定硬编码
# ==========================================

# ----------------- A. 相机内参与畸变 -----------------
# Cam0 (左目)
M1 = np.array([
    [1779.306630528, 0.0,            1236.495890436],
    [0.0,            1779.808565553, 1028.704529665],
    [0.0,            0.0,            1.0           ]
])
d1 = np.array([-0.061257288, 0.058743808, 0.000775611, 0.000709245])

# Cam1 (右目)
M2 = np.array([
    [1778.180341291, 0.0,            1238.311818044],
    [0.0,            1779.141167209, 1003.565528323],
    [0.0,            0.0,            1.0           ]
])
d2 = np.array([-0.063398165, 0.060703059, -0.000425742, -0.000270349])


# ----------------- B. 空间外参矩阵 (单位：米 -> 毫米) -----------------
# 1. 双目外参 T_c1_c0 (Cam0 -> Cam1, 用于立体极线校正)
# 对应 Kalibr 报告中的 Baseline (cam0 to cam1)
T_cam0_to_cam1_meters = np.array([
    [ 0.99389104,  0.02924261,  0.10642123, -0.09627463],
    [-0.03149356,  0.99931309,  0.01953227,  0.00266601],
    [-0.10577695, -0.02276453,  0.99412927,  0.00857531],
    [ 0.0,         0.0,         0.0,         1.0       ]
])
T_cam0_to_cam1_mm = T_cam0_to_cam1_meters.copy()
T_cam0_to_cam1_mm[:3, 3] *= 1000.0  # 平移向量单位转为 mm

# 2. 视惯外参 T_ic (Cam0 -> IMU, 用于把局部点云挂载到IMU轨迹上)
T_cam0_to_imu_meters = np.array([
    [-0.99971903, -0.01413073, -0.01903129,  0.00066   ],
    [ 0.01894373,  0.00630457, -0.99980067, -0.00207161],
    [ 0.0142479,  -0.99988028, -0.0060351,   0.00068844],
    [ 0.0,         0.0,         0.0,         1.0       ]
])
T_cam0_to_imu_mm = T_cam0_to_imu_meters.copy()
T_cam0_to_imu_mm[:3, 3] *= 1000.0  # 平移向量单位转为 mm


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

    # LUT 静态映射表构建: 提前算好虚拟图像到真实图像的亚像素坐标偏移量 (map1x, map1y)，以节约算力。
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


def compute_sgm_disparity(rectL, rectR, min_disp, num_disp, block_size, multiplier, max_diff, uniqueness):
    P1 = ACTUAL_P1
    P2 = ACTUAL_P2

    # blockSize(匹配块大小)：控制感受野大小；过大导致物体边缘膨胀且细节丢失，过小抗噪能力弱且易产生弱纹理空洞。
    #
    # P1 & P2(平滑惩罚项)：控制表面深度连续性；过大容易抹平真实的深度突变（粘连边缘），过小无法填补弱纹理区域的空洞。
    #
    # uniquenessRatio(唯一性比率)：控制匹配结果的置信度门槛；过大（严苛）会导致弱纹理区大量空洞，过小（宽容）会引入错误匹配的杂色噪点。
    #
    # disp12MaxDiff(左右一致性校验)：控制左右眼视差结果的容忍度；过小（严苛）边缘干净但易产生闭塞空洞，过大（宽松）能保住弱纹理但会在物体边缘产生拉丝 / 飞线。
    #
    # minDisparity & numDisparities(视差搜索范围)：控制有效重构的物理距离范围；设置不当会导致目标物体超出范围而被直接截断或完全丢失。
    #
    # speckleWindowSize & speckleRange(散斑滤波)：控制悬浮孤立噪点的剔除力度；窗口过大或范围过小会误删真实的细小结构，反之则无法有效清除空中飞点。
    #
    # mode(动态规划模式)：控制全局平滑的扫描方向；MODE_HH（8方向）平滑效果最佳但速度极慢，MODE_SGBM（默认5方向）速度较快且平滑效果适中。

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size,
        P1=P1, P2=P2,
        disp12MaxDiff=max_diff, uniquenessRatio=uniqueness,
        speckleWindowSize=1000, speckleRange=2, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH
    )

    disparity_left = left_matcher.compute(rectL, rectR)
    return disparity_left.astype(np.float32) / 16.0


def generate_local_pcd(disparity, gray_img, Q):
    grad_x = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    EDGE_GRADIENT_THRESHOLD = 6.0

    points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True).reshape(-1, 3)
    disp_flat = disparity.reshape(-1)
    gray_flat = gray_img.reshape(-1)
    grad_flat = grad_mag.reshape(-1)

    mask = (
            (np.isfinite(points_3d[:, 0])) &
            (disp_flat > 0) &
            (points_3d[:, 2] > MIN_Z_DIST) &
            (points_3d[:, 2] < MAX_Z_DIST) &
            (grad_flat < EDGE_GRADIENT_THRESHOLD) &
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
# 4. 主干流水线
# ==========================================
def main():
    os.makedirs(OUTPUT_LOCAL_DIR, exist_ok=True)
    df_traj = load_vins_trajectory(VIO_CSV_PATH)

    # 1. 预加载并解析所有左目图像的时间戳，建立时间轴池
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

    # 初始化矩阵与全局SGM参数
    sample_img = cv2.imread(valid_img_paths[0], 0)
    rect_maps = get_rectification_maps((sample_img.shape[1], sample_img.shape[0]))
    Q, R1 = rect_maps['Q'], rect_maps['R1']

    # 手动设置视差参数MIN_DISP和NUM_DISP
    GLOBAL_MIN_DISP = MANUAL_MIN_DISP
    GLOBAL_NUM_DISP = MANUAL_NUM_DISP
    print(f"\n[INFO] 视差参数: minDisp={GLOBAL_MIN_DISP}, numDisp={GLOBAL_NUM_DISP}")


    chunk_pcds = []  # 【新增】局部点云缓冲池列表
    chunk_index = 0
    processed_count = 0

    print(f"\n[START] 开始基于位姿主导的点云拼接！")
    print(f"        -> 有效 VIO 位姿总数: {len(df_traj)}")
    print(f"        -> VIO 采样间隔: {VIO_INTERVAL}")

    # ================== 【核心逻辑修改】 ==================
    # 遍历 VIO 轨迹，用轨迹去寻找最接近的图像
    for idx in range(0, len(df_traj), VIO_INTERVAL):
        row = df_traj.iloc[idx]
        pose_ts = row['timestamp']

        # 寻找与当前位姿时间戳最接近的图像
        time_diffs = np.abs(img_timestamps - pose_ts)
        closest_img_idx = np.argmin(time_diffs)
        min_time_diff = time_diffs[closest_img_idx]

        # 严格的时间容差：由于你的图像流是 20fps(50ms间隔)，这里设为 0.03 秒(30ms)极其安全
        if min_time_diff > 0.03:
            print(f"[WARN] VIO 位姿 (ts:{pose_ts:.3f}) 找不到匹配的图像 (最小误差:{min_time_diff:.3f}s)，已跳过。")
            continue

        imgL_path = valid_img_paths[closest_img_idx]
        img_name = os.path.basename(imgL_path)
        imgR_path = os.path.join(RIGHT_IMG_DIR, img_name)

        if not os.path.exists(imgR_path): continue

        print(f"[PROCESS] 提取位姿 (idx:{idx}) -> 匹配图像: {img_name} (时间差: {min_time_diff * 1000:.1f} ms)")

        # 读取与校正
        imgL = cv2.imread(imgL_path, 0)
        imgR = cv2.imread(imgR_path, 0)
        rectL = cv2.remap(imgL, rect_maps['map1x'], rect_maps['map1y'], cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, rect_maps['map2x'], rect_maps['map2y'], cv2.INTER_LINEAR)

        # 深度图与局部点云生成
        disp = compute_sgm_disparity(rectL, rectR, GLOBAL_MIN_DISP, GLOBAL_NUM_DISP, SGM_BLOCK_SIZE, SGM_MULTIPLIER, SGM_MAX_DIFF, SGM_UNIQUENESS)
        local_pcd = generate_local_pcd(disp, rectL, Q)
        if len(local_pcd.points) == 0: continue

        local_pcd = local_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        # 局部ROR滤波
        # cl, ind = local_pcd.remove_radius_outlier(nb_points=ROR_NB_POINTS, radius=ROR_RADIUS)
        # local_pcd = local_pcd.select_by_index(ind)

        # 坐标系空间刚体变换
        T_rect0_to_raw0 = np.eye(4)
        T_rect0_to_raw0[:3, :3] = R1.T
        T_wi = np.eye(4)
        r = R.from_quat([row['q_x'], row['q_y'], row['q_z'], row['q_w']])
        T_wi[:3, :3] = r.as_matrix()
        T_wi[:3, 3] = [row['p_x'] * 1000.0, row['p_y'] * 1000.0, row['p_z'] * 1000.0]

        T_wi_ic = np.dot(T_wi, T_cam0_to_imu_mm)
        T_final = np.dot(T_wi_ic, T_rect0_to_raw0)
        local_pcd.transform(T_final)

        chunk_pcds.append(local_pcd)  # 先不拼入全局，存入缓冲池
        processed_count += 1

        # ==========================================================
        # 【内存管理】分块合并与下采样 (Chunking Strategy)
        # ==========================================================
        if len(chunk_pcds) >= 15:  # 每攒够 X 帧处理一次区块
            # 1. 瞬间合并列表里的所有点云 (内部指针操作，效率极高)
            merged_chunk = chunk_pcds[0]
            for p in chunk_pcds[1:]:
                merged_chunk += p

            # 2. 对这 X 帧形成的局部区块进行独立下采样
            merged_chunk = merged_chunk.voxel_down_sample(voxel_size=VOXEL_SIZE)

            # 3. 写入硬盘
            chunk_filename = os.path.join(OUTPUT_LOCAL_DIR, f"chunk_{chunk_index:03d}.ply")
            o3d.io.write_point_cloud(chunk_filename, merged_chunk)
            print(f"    -> [硬盘缓存] 已保存局部区块: chunk_{chunk_index:03d}.ply (包含 {len(merged_chunk.points)} 个点)")

            # 4. 清空列表，彻底释放这 X 帧原始高密数据的内存！
            chunk_pcds.clear()
            chunk_index += 1



    # ==========================================================
    # 【缓冲池收尾】合并缓冲池里剩下的不足 X 帧的点云
    # ==========================================================
    if len(chunk_pcds) > 0:
        merged_chunk = chunk_pcds[0]
        for p in chunk_pcds[1:]:
            merged_chunk += p
        merged_chunk = merged_chunk.voxel_down_sample(voxel_size=VOXEL_SIZE)
        chunk_filename = os.path.join(OUTPUT_LOCAL_DIR, f"chunk_{chunk_index:03d}.ply")
        o3d.io.write_point_cloud(chunk_filename, merged_chunk)
        print(f"    -> [硬盘缓存] 尾部残余区块已保存: chunk_{chunk_index:03d}.ply")
        chunk_pcds.clear()

    # ================== 阶段性完成提示 ==================
    print(f"\n[FINISH] Python 阶段处理完毕！共处理了 {processed_count} 帧具有强几何位移的局部点云。")
    print(f"[SUCCESS] 所有局部区块点云 (单位: mm) 已保存在目录: {OUTPUT_LOCAL_DIR}")
    print(f"💡 【下一步指示】: ")
    print(f"    1. 打开 CloudCompare。")
    print(f"    2. 将 {OUTPUT_LOCAL_DIR} 文件夹下的所有 chunk_xxx.ply 全选并拖入。")
    print(f"    3. 全选图层，点击 Edit -> Merge 一键合并为全局模型。")
    print(f"    4. 使用 CC 自带的 Subsample 和 SOR 滤波进行最终的降采样与去噪。")



if __name__ == "__main__":
    main()