import cv2
import numpy as np
import os
import glob
import open3d as o3d

# ==========================================
# 1. 核心运行与文件路径配置及参数定义
# ==========================================
DATASET_DIR = "./capture_and_calib/SGM_dataset/plate_static7.17"  # 数据集根目录
LEFT_IMG_DIR = os.path.join(DATASET_DIR, "cam0")
RIGHT_IMG_DIR = os.path.join(DATASET_DIR, "cam1")
OUTPUT_LOCAL_DIR = os.path.join(DATASET_DIR, "local_pcds")

# ----------------- 【SGBM 实验核心参数 (用于控制变量实验)】 -----------------
SGM_BLOCK_SIZE = 11
SGM_MULTIPLIER = 1  # 决定 P1 和 P2 的平滑惩罚倍数 (建议测试: 1, 2, 5, 8)
SGM_UNIQUENESS = 15  # 置信度/唯一性比率 (建议测试: 3, 5, 10, 15)
SGM_MAX_DIFF = 2  # 左右一致性校验容差 (建议测试: 2, 5, 32, 64)

# ================== 【手动视差范围参数】 ==================
MANUAL_MIN_DISP = 486  # 最小视差，对应最远工作距离
MANUAL_NUM_DISP = 128  # 视差搜索范围，值越大能看清越近的物体

# 算出真实的 P1 和 P2 数值
ACTUAL_P1 = 4 * SGM_MULTIPLIER * SGM_BLOCK_SIZE ** 2
ACTUAL_P2 = 16 * SGM_MULTIPLIER * SGM_BLOCK_SIZE ** 2

# ----------------- 【物理空间裁剪参数】 -----------------
MIN_Z_DIST = 100.0  # 最小有效距离 (mm)
MAX_Z_DIST = 500.0  # 最大有效距离 (mm)

# ----------------- 【体素下采样参数】 -----------------
VOXEL_SIZE_LOCAL = 0.05  # 局部点云下采样体素 (mm)

print(f"[INIT] 动态配置加载完毕，准备执行单帧 SGM 解算。")

# ==========================================
# 2. 硬件标定硬编码highres (2026-07-15 最新标定)
# ==========================================

# ----------------- A. 相机内参与畸变 -----------------
# Cam0 (左目)
M1 = np.array([
    [1779.306630528, 0.0, 1236.495890436],
    [0.0, 1779.808565553, 1028.704529665],
    [0.0, 0.0, 1.0]
])
d1 = np.array([-0.061257288, 0.058743808, 0.000775611, 0.000709245])

# Cam1 (右目)
M2 = np.array([
    [1778.180341291, 0.0, 1238.311818044],
    [0.0, 1779.141167209, 1003.565528323],
    [0.0, 0.0, 1.0]
])
d2 = np.array([-0.063398165, 0.060703059, -0.000425742, -0.000270349])

# ----------------- B. 空间外参矩阵 (单位：米 -> 毫米) -----------------
# 双目外参 T_c1_c0 (Cam0 -> Cam1, 用于立体极线校正)
T_cam0_to_cam1_meters = np.array([
    [0.99389104, 0.02924261, 0.10642123, -0.09627463],
    [-0.03149356, 0.99931309, 0.01953227, 0.00266601],
    [-0.10577695, -0.02276453, 0.99412927, 0.00857531],
    [0.0, 0.0, 0.0, 1.0]
])
T_cam0_to_cam1_mm = T_cam0_to_cam1_meters.copy()
T_cam0_to_cam1_mm[:3, 3] *= 1000.0  # 平移向量单位转为 mm


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


def compute_sgm_disparity(rectL, rectR, min_disp, num_disp, block_size, multiplier, max_diff, uniqueness):
    P1 = ACTUAL_P1
    P2 = ACTUAL_P2

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
    EDGE_GRADIENT_THRESHOLD = 8.0

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
# 4. 主干流水线 (纯视觉单帧遍历)
# ==========================================
def main():
    os.makedirs(OUTPUT_LOCAL_DIR, exist_ok=True)

    # 预加载所有左目图像
    print("[INFO] 正在扫描图像序列...")
    left_images_raw = sorted(
        glob.glob(os.path.join(LEFT_IMG_DIR, "*.png")) + glob.glob(os.path.join(LEFT_IMG_DIR, "*.bmp")))

    if not left_images_raw:
        print("[ERROR] 没有找到左目图像文件！请检查路径。")
        return

    # 初始化矩阵与校正映射表
    sample_img = cv2.imread(left_images_raw[0], 0)
    rect_maps = get_rectification_maps((sample_img.shape[1], sample_img.shape[0]))
    Q, R1 = rect_maps['Q'], rect_maps['R1']

    GLOBAL_MIN_DISP = MANUAL_MIN_DISP
    GLOBAL_NUM_DISP = MANUAL_NUM_DISP
    print(f"\n[INFO] 视差参数: minDisp={GLOBAL_MIN_DISP}, numDisp={GLOBAL_NUM_DISP}")
    print(f"\n[START] 开始基于图像时间轴的点云解算并自动保存！")

    processed_count = 0

    # 遍历左目图像文件夹，执行配对解算
    for imgL_path in left_images_raw:
        img_name = os.path.basename(imgL_path)
        imgR_path = os.path.join(RIGHT_IMG_DIR, img_name)

        if not os.path.exists(imgR_path):
            print(f"[WARN] 找不到匹配的右目图像: {img_name}，已跳过。")
            continue

        print(f"[PROCESS] 正在解算: {img_name} ...")

        # 读取与校正
        imgL = cv2.imread(imgL_path, 0)
        imgR = cv2.imread(imgR_path, 0)
        rectL = cv2.remap(imgL, rect_maps['map1x'], rect_maps['map1y'], cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, rect_maps['map2x'], rect_maps['map2y'], cv2.INTER_LINEAR)

        # 深度图与局部点云生成
        disp = compute_sgm_disparity(rectL, rectR, GLOBAL_MIN_DISP, GLOBAL_NUM_DISP, SGM_BLOCK_SIZE, SGM_MULTIPLIER,
                                     SGM_MAX_DIFF, SGM_UNIQUENESS)
        local_pcd = generate_local_pcd(disp, rectL, Q)

        if len(local_pcd.points) == 0:
            print(f"    -> [SKIP] 有效三维点数为 0，跳过保存。")
            continue

        # 局部体素降采样
        local_pcd = local_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_LOCAL)

        # 坐标系空间刚体变换 (将极大减少后续在 CloudCompare 里对齐的麻烦)
        # 点云原本位于“校正后的水平极线相机系”，通过 R1.T 转回“原始左相机物理坐标系”
        T_rect0_to_raw0 = np.eye(4)
        T_rect0_to_raw0[:3, :3] = R1.T
        local_pcd.transform(T_rect0_to_raw0)

        # 动态命名与本地保存
        # 把原图拓展名（如 .png / .bmp）替换为 .ply 格式存储
        local_pcd_name = os.path.splitext(img_name)[0] + '.ply'
        local_save_path = os.path.join(OUTPUT_LOCAL_DIR, local_pcd_name)
        o3d.io.write_point_cloud(local_save_path, local_pcd)

        processed_count += 1

    print(f"\n[FINISH] 任务完毕！")
    print(f"         共成功解算并保存了 {processed_count} 帧局部点云。")
    print(f"         输出路径: {OUTPUT_LOCAL_DIR}")


if __name__ == "__main__":
    main()