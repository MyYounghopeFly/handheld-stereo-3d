import cv2
import os
import glob
import numpy as np

# =================================================================
# 1. 核心路径定义与性能参数配置
# =================================================================
DATASET_DIR = "C:/Users/12939/Desktop/Structured-light-stereo-main/capture_and_calib/SGM_dataset/steel7.12"  # 数据集根目录
LEFT_IMG_DIR = os.path.join(DATASET_DIR, "cam0")  # 左目图像文件夹
RIGHT_IMG_DIR = os.path.join(DATASET_DIR, "cam1")  # 右目图像文件夹
OUTPUT_VIDEO_PATH = os.path.join(DATASET_DIR, "stereo_stream.mp4") # 输出视频路径 (MP4格式)

# ----------------- 【视频控制参数】 -----------------
# 你的相机原图是 20fps。
# 如果想原速播放，设为 20；如果想慢动作仔细看弱纹理，可以设为 10
VIDEO_FPS = 10
RESIZE_FACTOR = 0.5  # 分辨率缩放系数 (0.5 能极大减小视频体积，极度适合微信发送)

# ----------------- 【显示模式选择】 -----------------
# 'side_by_side' : 左右目并排拼合
# 'left_only'     : 仅将左目图像流转为视频
# 'right_only'    : 仅将右目图像流转为视频
VIEW_MODE = 'side_by_side'


# =================================================================
# 2. 图像读取与基于时间戳的严格排序
# =================================================================
def create_stereo_video():
    print(f"[INIT] 开始准备 MP4 视频转换生成器...")
    print(f" -> 模式: {VIEW_MODE} | 目标帧率: {VIDEO_FPS} FPS | 缩放系数: {RESIZE_FACTOR}")

    # 支持 png 和 bmp 格式
    search_path = os.path.join(LEFT_IMG_DIR, "*.[pP][nN][gG]")
    left_img_paths = sorted(glob.glob(search_path))

    if not left_img_paths:
        search_path = os.path.join(LEFT_IMG_DIR, "*.[bB][mM][pP]")
        left_img_paths = sorted(glob.glob(search_path))

    if not left_img_paths:
        print(f"[ERROR] 错误：未能在目录 {LEFT_IMG_DIR} 中找到任何图像文件！")
        return

    # 基于时间戳数值排序
    try:
        left_img_paths = sorted(left_img_paths, key=lambda x: float(os.path.basename(x).split('.')[0]))
    except ValueError:
        print("[WARN] 警告：文件名未完全符合数字时间戳格式，退回默认字符串排序。")
        left_img_paths = sorted(left_img_paths)

    total_found = len(left_img_paths)
    print(f"[INFO] 成功检索到左目图像共计: {total_found} 帧。")

    # =================================================================
    # 3. 预读取第一帧，初始化 OpenCV 视频写入器 (VideoWriter)
    # =================================================================
    # 视频编码器必须提前知道每一帧的精确分辨率
    first_imgL = cv2.imread(left_img_paths[0])
    if first_imgL is None:
        print("[ERROR] 第一帧读取失败，请检查路径！")
        return

    if VIEW_MODE == 'side_by_side':
        # 并排模式宽度翻倍
        base_h, base_w = first_imgL.shape[:2]
        target_w = int(base_w * 2 * RESIZE_FACTOR)
        target_h = int(base_h * RESIZE_FACTOR)
    else:
        base_h, base_w = first_imgL.shape[:2]
        target_w = int(base_w * RESIZE_FACTOR)
        target_h = int(base_h * RESIZE_FACTOR)

    print(f"[INFO] 视频最终输出分辨率将设为: {target_w}x{target_h}")

    # 定义 mp4v 编码器 (兼容性最好的 MP4 编码之一)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, VIDEO_FPS, (target_w, target_h))

    # =================================================================
    # 4. 图像处理与画幅拼合主循环
    # =================================================================
    print(f"[WRITE] 正在逐帧压制视频，请稍候...")

    for idx, left_path in enumerate(left_img_paths):
        img_name = os.path.basename(left_path)
        right_path = os.path.join(RIGHT_IMG_DIR, img_name)

        imgL = cv2.imread(left_path)
        if imgL is None: continue

        # 模式处理
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

        # 缩放处理
        if RESIZE_FACTOR != 1.0:
            frame_out = cv2.resize(frame_out, (target_w, target_h))

        # 写入视频流
        video_writer.write(frame_out)

        if (idx + 1) % 50 == 0:
            print(f"   ... 已处理 {idx + 1}/{total_found} 帧 ...")

    # =================================================================
    # 5. 释放资源并完成写入
    # =================================================================
    video_writer.release()
    print(f"\n🎉 [SUCCESS] MP4 视频渲染成功！")
    print(f"💾 最终保存路径: {OUTPUT_VIDEO_PATH}")
    print(f"📱 提示：现在你可以直接把这个 .mp4 文件发到微信里，无缝流畅播放了！")


if __name__ == "__main__":
    create_stereo_video()