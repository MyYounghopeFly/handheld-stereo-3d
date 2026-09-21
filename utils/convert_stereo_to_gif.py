import cv2
import os
import glob
import numpy as np
from PIL import Image

# =================================================================
# 1. 核心路径定义与性能参数配置
# =================================================================
DATASET_DIR = "C:/Users/12939/Desktop/Structured-light-stereo-main/capture_and_calib/SGM_dataset/steel_drum7.2"
LEFT_IMG_DIR = os.path.join(DATASET_DIR, "cam0")
RIGHT_IMG_DIR = os.path.join(DATASET_DIR, "cam1")
OUTPUT_GIF_PATH = os.path.join(DATASET_DIR, "stereo_stream.gif")

# ----------------- 图像与输出参数 -----------------
VIDEO_FPS = 10
RESIZE_FACTOR = 0.20
VIEW_MODE = 'side_by_side'
FRAME_STEP = 10  # 抽帧步长设定：例如设为3，则每隔3帧提取1帧


# =================================================================
# 2. 核心转换逻辑
# =================================================================
def create_stereo_gif():
    print(f"[INFO] 初始化 GIF 转换程序...")
    print(f"[INFO] 视图模式: {VIEW_MODE}, 目标帧率: {VIDEO_FPS} FPS, 抽帧步长: {FRAME_STEP}")

    search_path = os.path.join(LEFT_IMG_DIR, "*.[pP][nN][gG]")
    left_img_paths = sorted(glob.glob(search_path))

    if not left_img_paths:
        search_path = os.path.join(LEFT_IMG_DIR, "*.[bB][mM][pP]")
        left_img_paths = sorted(glob.glob(search_path))

    if not left_img_paths:
        print(f"[ERROR] 未能在目录 {LEFT_IMG_DIR} 中检索到图像文件。")
        return

    try:
        left_img_paths = sorted(left_img_paths, key=lambda x: float(os.path.basename(x).split('.')[0]))
    except ValueError:
        print("[WARN] 文件名未完全符合数字时间戳格式，退回默认字符串排序。")
        left_img_paths = sorted(left_img_paths)

    # 核心抽帧逻辑：通过列表切片实现全局间隔采样
    if FRAME_STEP > 1:
        left_img_paths = left_img_paths[::FRAME_STEP]
        print(f"[INFO] 已应用全局抽帧，等距采样步长为 {FRAME_STEP}。")

    total_found = len(left_img_paths)
    print(f"[INFO] 共计进入处理队列的图像: {total_found} 帧。")

    # =================================================================
    # 3. 图像遍历与格式转换
    # =================================================================
    frames_list = []

    for idx, left_path in enumerate(left_img_paths):
        img_name = os.path.basename(left_path)
        right_path = os.path.join(RIGHT_IMG_DIR, img_name)

        imgL = cv2.imread(left_path)
        if imgL is None:
            continue

        if VIEW_MODE == 'left_only':
            frame_out = imgL
        elif VIEW_MODE == 'right_only':
            imgR = cv2.imread(right_path)
            frame_out = imgR if imgR is not None else np.zeros_like(imgL)
        elif VIEW_MODE == 'side_by_side':
            imgR = cv2.imread(right_path)
            if imgR is None:
                imgR = np.zeros_like(imgL)
            if imgL.shape != imgR.shape:
                imgR = cv2.resize(imgR, (imgL.shape[1], imgL.shape[0]))
            frame_out = cv2.hconcat([imgL, imgR])

        if RESIZE_FACTOR != 1.0:
            target_w = int(frame_out.shape[1] * RESIZE_FACTOR)
            target_h = int(frame_out.shape[0] * RESIZE_FACTOR)
            frame_out = cv2.resize(frame_out, (target_w, target_h))

        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames_list.append(pil_img)

        if (idx + 1) % 20 == 0:
            print(f"[INFO] 渲染进度: {idx + 1}/{total_found} 帧")

    # =================================================================
    # 4. GIF 文件写入
    # =================================================================
    if frames_list:
        print(f"[INFO] 正在将帧序列写入硬盘，请等待...")
        duration_ms = int(1000 / VIDEO_FPS)

        frames_list[0].save(
            OUTPUT_GIF_PATH,
            save_all=True,
            append_images=frames_list[1:],
            optimize=True,
            duration=duration_ms,
            loop=0
        )
        print(f"[INFO] 渲染任务结束。输出路径: {OUTPUT_GIF_PATH}")
    else:
        print(f"[ERROR] 图像序列为空，渲染终止。")


if __name__ == "__main__":
    create_stereo_gif()