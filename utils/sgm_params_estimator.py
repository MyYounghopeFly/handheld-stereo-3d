import cv2
import numpy as np
import os
import glob
import math

# =========================================================
# 1. 参数与路径配置 (保持与您提供的一致)
# =========================================================
LEFT_IMG_DIR = r"C:\Users\12939\Desktop\Structured-light-stereo-main\capture_and_calib\SGM_dataset\disp_calib_steel_plate7.16\cam0"
RIGHT_IMG_DIR = r"C:\Users\12939\Desktop\Structured-light-stereo-main\capture_and_calib\SGM_dataset\disp_calib_steel_plate7.16\cam1"

# 棋盘格规格 (注意：这里指的是内部角点的 列数 x 行数)
# 如果您的 12x9 是指方块数量，请将此处改为 (11, 8)
BOARD_SIZE = (11, 8)

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

# 双目外参 T_c1_c0 (Cam0 -> Cam1)
T_cam0_to_cam1_meters = np.array([
    [0.99389104, 0.02924261, 0.10642123, -0.09627463],
    [-0.03149356, 0.99931309, 0.01953227, 0.00266601],
    [-0.10577695, -0.02276453, 0.99412927, 0.00857531],
    [0.0, 0.0, 0.0, 1.0]
])
T_cam0_to_cam1_mm = T_cam0_to_cam1_meters.copy()
T_cam0_to_cam1_mm[:3, 3] *= 1000.0  # 平移向量单位转为 mm
# 提取旋转矩阵R和平移向量T
R = T_cam0_to_cam1_mm[:3, :3]
T = T_cam0_to_cam1_mm[:3, 3].reshape(3, 1)


def main():
    # ---------------------------------------------------------
    # 1. 获取图像尺寸并计算极线校正映射
    # ---------------------------------------------------------
    left_images = sorted(
        glob.glob(os.path.join(LEFT_IMG_DIR, '*.png')) + glob.glob(os.path.join(LEFT_IMG_DIR, '*.jpg')))
    right_images = sorted(
        glob.glob(os.path.join(RIGHT_IMG_DIR, '*.png')) + glob.glob(os.path.join(RIGHT_IMG_DIR, '*.jpg')))

    if not left_images or not right_images:
        print("未找到图像，请检查路径！")
        return

    sample_img = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
    img_size = (sample_img.shape[1], sample_img.shape[0])  # (width, height)
    print(f"检测到图像分辨率: {img_size[0]}x{img_size[1]}")

    # 计算极线校正参数
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        M1, d1, M2, d2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # 生成校正映射表 (Map)
    map1x, map1y = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_size, cv2.CV_32FC1)
    print("已成功生成极线校正映射表。")

    all_disparities = []

    # 亚像素角点优化的终止条件：最大迭代次数30次，或精度达到0.001
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # ---------------------------------------------------------
    # 2. 遍历图像对，校正并提取棋盘格角点视差
    # ---------------------------------------------------------
    print(f"开始处理 {len(left_images)} 对标定板图像提取视差...")
    for left_path, right_path in zip(left_images, right_images):
        imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        # 步骤 A: 应用极线校正，消除镜头畸变并将图像对齐到同一水平极线
        rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)


        # 步骤 B: 寻找棋盘格角点
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        retL, cornersL = cv2.findChessboardCorners(rectL, BOARD_SIZE, flags)
        retR, cornersR = cv2.findChessboardCorners(rectR, BOARD_SIZE, flags)

        # 如果左右图都成功找到了完整的棋盘格角点
        if retL and retR:
            # 步骤 C: 亚像素级精确化角点位置，极大提高测量精度
            cornersL = cv2.cornerSubPix(rectL, cornersL, (11, 11), (-1, -1), subpix_criteria)
            cornersR = cv2.cornerSubPix(rectR, cornersR, (11, 11), (-1, -1), subpix_criteria)

            y_errors = []  # 用于记录当前图像对的 Y 误差

            # 步骤 D: 计算每一个对应角点的视差
            # OpenCV 的 findChessboardCorners 返回的角点顺序是一致的
            for ptL, ptR in zip(cornersL, cornersR):
                x_l, y_l = ptL[0]
                x_r, y_r = ptR[0]

                y_err = abs(y_l - y_r)
                y_errors.append(y_err)

                # 暂时放宽条件，或者干脆不设限，先把视差存下来
                # 哪怕 Y 误差很大，我们也强行算一下 X 视差看看
                disp = x_l - x_r
                all_disparities.append(disp)

            avg_y_err = np.mean(y_errors)
            max_y_err = np.max(y_errors)
            print(f"图像 {os.path.basename(left_path)} -> Y 轴平均误差: {avg_y_err:.2f} 像素, 最大误差: {max_y_err:.2f} 像素")


        else:
            print(f"警告: 图像对 {os.path.basename(left_path)} 未能提取出完整的 {BOARD_SIZE} 棋盘格，已跳过。")

    # ---------------------------------------------------------
    # 3. 统计结果并估算 SGM 参数
    # ---------------------------------------------------------
    if not all_disparities:
        print("\n错误：未能提取到任何有效的视差数据！")
        return

    min_actual_disp = min(all_disparities)
    max_actual_disp = max(all_disparities)
    avg_actual_disp = np.mean(all_disparities)

    print("\n" + "=" * 40)
    print("            测量统计结果")
    print("=" * 40)
    print(f"有效角点样本总数: {len(all_disparities)} 个")
    print(f"测得最小视差: {min_actual_disp:.2f} 像素")
    print(f"测得最大视差: {max_actual_disp:.2f} 像素")
    print(f"测得平均视差: {avg_actual_disp:.2f} 像素")

    # --- 计算 SGBM 需要的参数 ---
    # 余量 margin：考虑到实际钢板的厚度公差、放置误差，向外扩充搜索范围
    margin = 16

    # MIN_DISP: 最小视差向下取整并减去余量
    min_disp = math.floor(min_actual_disp) - margin
    # 绝大部分前向平行双目系统视差是正数，兜底防止负数
    min_disp = max(0, min_disp)

    # NUM_DISP: 视差范围，必须是 16 的整数倍
    max_disp = math.ceil(max_actual_disp) + margin
    num_disp = max_disp - min_disp

    if num_disp % 16 != 0:
        num_disp = ((num_disp // 16) + 1) * 16

    print("\n" + "=" * 40)
    print("         SGM/SGBM 推荐配置参数")
    print("=" * 40)
    print(f"建议 MIN_DISP   = {min_disp}")
    print(f"建议 NUM_DISP   = {num_disp}")
    print(f"理论搜索范围    = [{min_disp}, {min_disp + num_disp}] 像素")
    print("(注: 已经包含了前后工差余量，并自动遵守了 NUM_DISP 为 16 的倍数规则)")


if __name__ == "__main__":
    main()