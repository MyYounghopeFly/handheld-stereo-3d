import cv2
import numpy as np
import os
import glob
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort
import copy

# ==========================================
# 核心运行与文件路径配置及参数定义
# ==========================================
DATASET_DIR = "./capture_and_calib/SGM_dataset/bottle9.14"  # 数据集根目录
MODEL_PATH = r"C:\Users\12939\Downloads\resources_iter10\crestereo_combined_iter10_480x640.onnx"  # 深度学习模型路径
LEFT_IMG_DIR = os.path.join(DATASET_DIR, "cam0")
RIGHT_IMG_DIR = os.path.join(DATASET_DIR, "cam1")
VIO_CSV_PATH = os.path.join(DATASET_DIR, "vio.csv")
BASE_LOCAL_DIR = os.path.join(DATASET_DIR, "local_pcds")

# ----------------- 【核心控制参数】 -----------------
# 设为 1 代表：把 vio.csv 里的每一帧有效位姿全部转化为点云糊上去（极限稠密度）
VIO_INTERVAL = 20

# ----------------- 【SGBM 实验核心参数 (用于控制变量实验)】 -----------------
SGM_BLOCK_SIZE = 25
SGM_MULTIPLIER = 1          # 决定 P1 和 P2 的平滑惩罚倍数 (建议测试: 1, 2, 5, 8)
SGM_UNIQUENESS = 25         # 置信度/唯一性比率 (建议测试: 3, 5, 10, 15)
SGM_MAX_DIFF = 1            # 左右一致性校验容差 (建议测试: 2, 5, 32, 64)

# ================== 【手动视差范围参数】 ==================
MANUAL_MIN_DISP = 486      # 最小视差，对应最远工作距离
MANUAL_NUM_DISP = 192      # 视差搜索范围，值越大能看清越近的物体

# 算出真实的 P1 和 P2 数值，以便写入文件名
ACTUAL_P1 = 8 * SGM_MULTIPLIER * SGM_BLOCK_SIZE ** 2
ACTUAL_P2 = 32 * SGM_MULTIPLIER * SGM_BLOCK_SIZE ** 2

# ----------------- 【物理空间裁剪参数】 -----------------
MIN_Z_DIST = 100.0  # 最小有效距离 (mm)
MAX_Z_DIST = 400.0  # 最大有效距离 (mm)

# ----------------- 【体素下采样参数】 -----------------
VOXEL_SIZE = 0.5  # 点云下采样体素 (mm)

# ----------------- 【去噪滤波参数】 -----------------
ROR_NB_POINTS = 100  # 半径内至少需要的邻居点数
ROR_RADIUS = 10.0  # 搜索半径 (mm)

SOR_NB_NEIGHBORS = 30  # 考察的邻居点数
SOR_STD_RATIO = 1.0  # 剔除标准差的阈值乘数

# ----------------- 【ICP 精配准参数】 -----------------
USE_ICP_REFINEMENT = True       # 是否在 VIO 粗变换后用 ICP 修正残余漂移
ICP_VOXEL_SIZE = 1.0            # 配准前的下采样体素（mm），太小则慢，太大则不准
ICP_MAX_CORR_DIST = 5.0         # ICP 最大对应点距离（mm），一般 = 点间距的 2~3 倍
ICP_MAX_ITER = 30               # ICP 最大迭代次数
ICP_FITNESS_THRESH = 0.05       # 配准质量阈值，低于此值认为配准失败，保留 VIO 结果
ICP_REF_MAX_POINTS = 500_000    # 参考点云上限，超过则做粗下采样，防止内存爆炸

# ----------------- 【RANSAC 平面剔除参数】 -----------------
ENABLE_PLANE_REMOVAL = True     # 是否启用平面剔除（去桌面/地面）
PLANE_DIST_THRESH = 3.0         # 平面内点距离阈值（mm），建议 2~3 倍 VOXEL_SIZE
PLANE_RANSAC_N = 3              # 拟合平面所需的最少点（固定 3）
PLANE_NUM_ITER = 2000           # RANSAC 迭代次数
PLANE_MIN_RATIO = 0.15          # 平面占比低于此值认为无显著平面，不做剔除

# ----------------- 【自动 ROI：DBSCAN 聚类参数】 -----------------
ENABLE_AUTO_ROI = True          # 是否启用自动 ROI（聚类保留最大簇）
ROI_DBSCAN_EPS = 5.0            # DBSCAN 邻域半径（mm），建议 5~10 倍 VOXEL_SIZE
ROI_DBSCAN_MIN_POINTS = 1000     # 最小簇点数，小于此视为噪声
ROI_KEEP_TOP_K = 1              # 保留点数最多的前 K 个簇（通常保留 1 个 = 物体本体）
ROI_MIN_RESULT_POINTS = 500     # ROI 剩余点数低于此值则视为检测失败，放弃本帧

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
# 基础函数与 SGM 算法模块
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


def generate_local_pcd(disparity, gray_img, Q,
                       min_z=MIN_Z_DIST, max_z=MAX_Z_DIST,
                       use_grad_filter=False, grad_thresh=6.0):
    # 1. 一次性反投影
    points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

    # 2. 全部展平
    points_flat = points_3d.reshape(-1, 3)
    disp_flat   = disparity.reshape(-1)
    gray_flat   = gray_img.reshape(-1)

    # 3. mask：便宜 → 贵的顺序
    mask = (
        (disp_flat > 0.5) &
        (points_flat[:, 2] > min_z) &
        (points_flat[:, 2] < max_z) &
        (gray_flat > 5) &
        (gray_flat < 250)
    )

    # 4. 可选的梯度滤波（只在需要时算）
    if use_grad_filter:
        gx = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy).reshape(-1)
        mask &= (grad_mag < grad_thresh)

    valid_points = points_flat[mask]
    if len(valid_points) == 0:
        return o3d.geometry.PointCloud()

    # 5. 颜色：float32 中间量 + repeat
    valid_gray = (gray_flat[mask].astype(np.float32) / 255.0).astype(np.float64)
    colors_rgb = np.repeat(valid_gray[:, None], 3, axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    return pcd

def remove_dominant_plane(pcd,
                          dist_thresh=PLANE_DIST_THRESH,
                          ransac_n=PLANE_RANSAC_N,
                          num_iter=PLANE_NUM_ITER,
                          min_ratio=PLANE_MIN_RATIO):
    """
    使用 RANSAC 检测并剔除点云中的主导平面（如桌面、地面）。
    返回剔除平面后的点云；若平面占比过低（无显著平面），则返回原始点云。
    """
    if len(pcd.points) < 100:
        return pcd

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=ransac_n,
        num_iterations=num_iter
    )

    plane_ratio = len(inliers) / len(pcd.points)
    if plane_ratio < min_ratio:
        # 平面不显著，不做剔除（避免误删物体）
        return pcd

    # 保留平面外的点（即物体）
    object_pcd = pcd.select_by_index(inliers, invert=True)
    return object_pcd

def auto_roi_by_clustering(pcd,
                           eps=ROI_DBSCAN_EPS,
                           min_points=ROI_DBSCAN_MIN_POINTS,
                           top_k=ROI_KEEP_TOP_K,
                           min_result_points=ROI_MIN_RESULT_POINTS):
    """
    通过 3D DBSCAN 聚类自动提取前景 ROI：保留点数最多的前 K 个簇。
    返回：(roi_pcd, n_clusters_total)
    - roi_pcd          : 裁剪出的 ROI 点云（若失败则返回空点云）
    - n_clusters_total : 检测到的有效簇数量
    """
    if len(pcd.points) < min_points:
        return o3d.geometry.PointCloud(), 0

    # DBSCAN 聚类
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    if labels.size == 0 or labels.max() < 0:
        return o3d.geometry.PointCloud(), 0

    # 统计每个有效簇的点数（labels == -1 表示噪声）
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return o3d.geometry.PointCloud(), 0

    # 按点数降序，保留 top_k 个簇
    sorted_idx = np.argsort(counts)[::-1]
    keep_labels = unique_labels[sorted_idx[:top_k]]

    keep_indices = np.where(np.isin(labels, keep_labels))[0]
    if len(keep_indices) < min_result_points:
        return o3d.geometry.PointCloud(), len(unique_labels)

    roi_pcd = pcd.select_by_index(keep_indices)
    return roi_pcd, len(unique_labels)

def refine_with_icp(source_pcd, reference_pcd,
                    voxel_size=ICP_VOXEL_SIZE,
                    max_corr_dist=ICP_MAX_CORR_DIST,
                    max_iter=ICP_MAX_ITER,
                    fitness_thresh=ICP_FITNESS_THRESH):
    """
    在 VIO 粗变换之后，用 ICP 把 source_pcd 精配准到 reference_pcd 上。
    返回：(refined_pcd, fitness, transform)
    - refined_pcd : 精配准后的点云（若配准失败，则返回原 source）
    - fitness     : 配准质量（0~1，越大越好）
    - transform   : 计算出的精配准矩阵
    """
    src_down = source_pcd.voxel_down_sample(voxel_size)
    tgt_down = reference_pcd.voxel_down_sample(voxel_size)

    if len(src_down.points) < 50 or len(tgt_down.points) < 50:
        return source_pcd, 0.0, np.eye(4)

    # VIO 已经给出了很好的初值，直接用单位矩阵作为 init 即可
    reg = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down,
        max_corr_dist,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )

    if reg.fitness < fitness_thresh:
        # 配准质量差，直接放弃，用 VIO 的原结果
        return source_pcd, reg.fitness, np.eye(4)

    refined = copy.deepcopy(source_pcd)
    refined.transform(reg.transformation)
    return refined, reg.fitness, reg.transformation


# ==============================================================================
# 深度学习立体匹配推理封装类 (基于 CREStereo ONNX)
# ==============================================================================
class CREStereoInfer:
    def __init__(self, onnx_model_path, input_shape=None, use_gpu=True):
        """
        :param onnx_model_path: 你的 .onnx 文件绝对路径
        :param input_shape: (Height, Width)，必须与你下载的 onnx 模型文件名保持严格一致！
                           例如 crestereo_combined_iter10_480x640.onnx 对应 (480, 640)
        """

        # 优先使用 GPU (CUDAExecutionProvider) 加速，若无环境自动回退到 CPU
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # sess_options.log_severity_level = 3  # 只输出 Error 级别，屏蔽 Warning/Info

        print(f"[AI-LOAD] 正在加载 CREStereo ONNX: {onnx_model_path}")
        self.session = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=providers)

        # 记录每个输入的 name 和 shape
        self.inputs_info = {inp.name: inp.shape for inp in self.session.get_inputs()}
        self.input_names = list(self.inputs_info.keys())
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

        # 打印，方便你确认
        for inp in self.session.get_inputs():
            print(f"[AI-LOAD] 输入: {inp.name}, shape: {inp.shape}")
        print(f"[AI-LOAD] 输出: {self.output_name}, shape: {self.output_shape}")

        # 从第一个输入的 shape 里取出 H, W
        # ONNX 的 shape 一般是 [N, C, H, W] 或带符号的 ['batch', 3, 240, 320]
        inp0 = self.session.get_inputs()[0]
        shape = inp0.shape
        print(f"[AI-LOAD] 模型输入 {inp0.name} 的 shape: {shape}")

        # 兼容 shape 里出现字符串/None 的情况
        h = shape[2]
        w = shape[3]
        if isinstance(h, int) and isinstance(w, int):
            self.net_h, self.net_w = h, w
            print(f"[AI-LOAD] 自动识别网络输入尺寸: {self.net_h} x {self.net_w}")
        else:
            # 模型是动态尺寸，退回到用户传入的值
            if input_shape is None:
                raise ValueError("模型是动态输入尺寸，请手动传入 input_shape")
            self.net_h, self.net_w = input_shape
            print(f"[AI-LOAD] 使用手动指定尺寸: {self.net_h} x {self.net_w}")

        print(f"[AI-LOAD] 当前执行后端: {self.session.get_providers()[0]}")


    def compute(self, rectL, rectR):
        """
        :param rectL, rectR: 已经过极线校正的左目和右目图像 (原始高分辨率，如 2448x2048)
        :return: 恢复到原始尺寸的 float32 绝对视差图 (像素单位)
        """
        orig_h, orig_w = rectL.shape[:2]

        # 1. 颜色通道转换 (CREStereo 网络基于 RGB 3通道训练)
        if len(rectL.shape) == 2:
            imgL_rgb = cv2.cvtColor(rectL, cv2.COLOR_GRAY2RGB)
            imgR_rgb = cv2.cvtColor(rectR, cv2.COLOR_GRAY2RGB)
        else:
            imgL_rgb = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)
            imgR_rgb = cv2.cvtColor(rectR, cv2.COLOR_BGR2RGB)

        # 2. 缩放到模型要求的网络输入尺寸 (480, 640)
        resized_L = cv2.resize(imgL_rgb, (self.net_w, self.net_h), interpolation=cv2.INTER_AREA)
        resized_R = cv2.resize(imgR_rgb, (self.net_w, self.net_h), interpolation=cv2.INTER_AREA)

        # 3. 归一化至 [0, 1] 并转换为 NCHW 张量格式: (1, 3, H, W)
        blob_L = (resized_L.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        blob_R = (resized_R.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        # 4. ONNX 前向推理

        # 为每个输入单独准备 blob（关键：每个输入的 H/W 可能不同！）
        input_feed = {}
        for name, shape in self.inputs_info.items():
            lname = name.lower()
            net_h, net_w = shape[2], shape[3]  # [N, C, H, W]

            if 'left' in lname:
                src = imgL_rgb
            elif 'right' in lname:
                src = imgR_rgb
            else:
                raise RuntimeError(f"无法识别的模型输入节点: {name}")

            resized = cv2.resize(src, (net_w, net_h), interpolation=cv2.INTER_AREA)
            blob = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
            input_feed[name] = blob


        outputs = self.session.run([self.output_name], input_feed)

        # 提取视差 (通常输出形状为 [1, 1, H, W] 或 [1, 2, H, W]，视差通常位于通道 0 或直接为 [H, W])
        raw_disp = np.squeeze(outputs[0])
        if raw_disp.ndim == 3:  # 如果包含 flow/disp 两个分量，取第0通道视差
            raw_disp = raw_disp[0]


        out_disp_h, out_disp_w = raw_disp.shape[:2]
        scale_factor = orig_w / float(out_disp_w)

        # 先乘缩放因子再插值回原图
        disp_full = cv2.resize(raw_disp * scale_factor,
                               (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)
        return disp_full.astype(np.float32)


# ==========================================
# 主干流水线
# ==========================================
def main():
    os.makedirs(OUTPUT_LOCAL_DIR, exist_ok=True)
    df_traj = load_vins_trajectory(VIO_CSV_PATH)

    ai_matcher = CREStereoInfer(MODEL_PATH, use_gpu=True)

    # 预加载并解析所有左目图像的时间戳，建立时间轴池
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
    icp_skipped = 0  # 【新增】ICP 配准失败的计数
    refinement_reference = None  # 【新增】ICP 精配准的参考点云（滚动更新）

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
        # disp = compute_sgm_disparity(rectL, rectR, GLOBAL_MIN_DISP, GLOBAL_NUM_DISP, SGM_BLOCK_SIZE, SGM_MULTIPLIER, SGM_MAX_DIFF, SGM_UNIQUENESS)
        disp = ai_matcher.compute(rectL, rectR)

        local_pcd = generate_local_pcd(disp, rectL, Q)
        if len(local_pcd.points) == 0: continue

        local_pcd = local_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        # 局部ROR滤波
        # cl, ind = local_pcd.remove_radius_outlier(nb_points=ROR_NB_POINTS, radius=ROR_RADIUS)
        # local_pcd = local_pcd.select_by_index(ind)

        # ============================================================
        # RANSAC 平面剔除（去桌面/背景）
        # ============================================================
        if ENABLE_PLANE_REMOVAL:
            before = len(local_pcd.points)
            local_pcd = remove_dominant_plane(local_pcd)
            after = len(local_pcd.points)
            if after == 0:
                # 整帧都被判定为平面（比如只看到桌面），丢弃
                continue
            # 只在点数减少明显时打印，避免刷屏
            if before - after > 500:
                print(f"    -> [RANSAC平面剔除] {before} -> {after} 点")

        # ============================================================
        # 【自动 ROI】DBSCAN 聚类，只保留最大的前景物体
        # ============================================================
        if ENABLE_AUTO_ROI:
            before = len(local_pcd.points)
            roi_pcd, n_clusters = auto_roi_by_clustering(local_pcd)
            if len(roi_pcd.points) < ROI_MIN_RESULT_POINTS:
                # 本帧未检测到有效前景（可能只看到桌面/空场景），跳过
                print(f"    -> [自动ROI] 未检测到有效前景 (总簇数:{n_clusters})，本帧跳过")
                continue
            local_pcd = roi_pcd
            after = len(local_pcd.points)
            print(f"    -> [自动ROI] {before} -> {after} 点 (共 {n_clusters} 个簇, 保留最大 {ROI_KEEP_TOP_K} 个)")

        # 坐标系空间刚体变换（VIO粗变换）
        T_rect0_to_raw0 = np.eye(4)
        T_rect0_to_raw0[:3, :3] = R1.T
        T_wi = np.eye(4)
        r = R.from_quat([row['q_x'], row['q_y'], row['q_z'], row['q_w']])
        T_wi[:3, :3] = r.as_matrix()
        T_wi[:3, 3] = [row['p_x'] * 1000.0, row['p_y'] * 1000.0, row['p_z'] * 1000.0]

        T_wi_ic = np.dot(T_wi, T_cam0_to_imu_mm)
        T_final = np.dot(T_wi_ic, T_rect0_to_raw0)
        local_pcd.transform(T_final)

        # ============================================================
        # ICP 精配准：修正 VIO 残余漂移
        # ============================================================
        if USE_ICP_REFINEMENT and refinement_reference is not None \
                and len(refinement_reference.points) > 0:
            local_pcd, fitness, _ = refine_with_icp(local_pcd, refinement_reference)
            if fitness < ICP_FITNESS_THRESH:
                icp_skipped += 1
                # 说明这次配准没做好，仍然保留 VIO 结果
                # 用原 VIO 变换后的结果（refine_with_icp 已返回原 pcd）

        # 更新滚动参考点云（用当前帧的细点云）
        if USE_ICP_REFINEMENT:
            if refinement_reference is None:
                refinement_reference = local_pcd.voxel_down_sample(ICP_VOXEL_SIZE)
            else:
                # Open3D 的 += 是引用式累加，先 clone 避免共享
                refinement_reference += local_pcd.voxel_down_sample(ICP_VOXEL_SIZE)
                # 防止参考点云无限膨胀
                if len(refinement_reference.points) > ICP_REF_MAX_POINTS:
                    refinement_reference = refinement_reference.voxel_down_sample(ICP_VOXEL_SIZE * 2)

        chunk_pcds.append(local_pcd)  # 先不拼入全局，存入缓冲池
        processed_count += 1

        # ==========================================================
        # 【内存管理】分块合并与下采样 (Chunking Strategy)
        # ==========================================================
        if len(chunk_pcds) >= 5:  # 每攒够 X 帧处理一次区块
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
    # 【缓冲池收尾】合并缓冲池里最后剩下的点云
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