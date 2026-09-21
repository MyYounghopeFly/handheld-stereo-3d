import os
import numpy as np
import cv2

if __name__ == '__main__':
    # 设置投影图案分辨率
    proj_w, proj_h = 912, 1140
    save_dir = 'patterns_912x1140_new'
    os.makedirs(save_dir, exist_ok=True)

    # 创建 GrayCodePattern 对象
    graycode = cv2.structured_light_GrayCodePattern.create(width=proj_w, height=proj_h)

    # 生成 Gray Code 图案
    retval, patterns = graycode.generate()
    if not retval:
        raise RuntimeError("Failed to generate GrayCode patterns.")

    # 生成黑白图用于遮挡区域检测
    black_img, white_img = graycode.getImagesForShadowMasks(
        np.zeros((proj_h, proj_w), dtype=np.uint8),
        np.zeros((proj_h, proj_w), dtype=np.uint8)
    )

    # 合并所有图像
    patterns = list(patterns)  # 转为列表方便追加
    patterns += [white_img, black_img]

    print(f"共生成图案：{len(patterns)} 张（包括横向、纵向、白图、黑图）")

    # 保存为 24-bit BMP 图像（3通道）
    for i, pattern in enumerate(patterns):
        filename = os.path.join(save_dir, f'pattern_{i:02d}.bmp')
        # 确保是 8-bit 单通道图像
        if pattern.dtype != np.uint8:
            pattern = (255 * pattern).astype(np.uint8)
        # 转换为三通道 BGR（OpenCV 默认格式）
        pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(filename, pattern_bgr)
        print(f"已保存：{filename}")
