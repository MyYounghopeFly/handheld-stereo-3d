import numpy as np
from PIL import Image
import os


def Generate_Speckle_Pattern(width, height, block_size=3, dots_per_block=2):
    """
    生成基于局部网格约束的随机二值散斑图案矩阵 (单通道)
    """
    print(f"正在生成 {width}x{height} 的基础散斑矩阵...")

    # 1. 初始化全黑矩阵 (单通道)
    # 使用 uint8 类型，0代表黑，255代表白
    image_matrix = np.zeros((height, width), dtype=np.uint8)

    # 2. 遍历图像：以 block_size (3) 为步长进行网格扫描
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):

            # 3. 边界检查
            if (y + block_size > height) or (x + block_size > width):
                continue

            # 4. 随机选择逻辑
            # 在 0 到 8 (3*3-1) 中无放回抽取 dots_per_block (2) 个位置
            total_pixels = block_size * block_size
            selected_indices = np.random.choice(
                range(total_pixels),
                dots_per_block,
                replace=False
            )

            # 5. 填充亮点
            for index in selected_indices:
                dr = index // block_size  # 行偏移
                dc = index % block_size  # 列偏移
                image_matrix[y + dr, x + dc] = 255

    return image_matrix


def main():
    # --- 参数配置 ---
    DLP_WIDTH = 912
    DLP_HEIGHT = 1140
    OUTPUT_FILENAME = "pattern_24bit_72dpi.bmp"

    # 1. 生成单通道散斑数据
    gray_data = Generate_Speckle_Pattern(DLP_WIDTH, DLP_HEIGHT)

    # 2. 转换为 24位 (RGB) 格式
    # DLP 要求 24位输入时，通常是 R=G=B 的灰度表现
    # 将 (H, W) 的数组堆叠成 (H, W, 3)
    rgb_data = np.stack((gray_data, gray_data, gray_data), axis=-1)

    # 3. 创建图像对象 (RGB模式即对应 24-bit 深度)
    img = Image.fromarray(rgb_data)

    # 4. 保存图像，指定 DPI
    print(f"正在保存为 24位 BMP, DPI=72...")
    img.save(OUTPUT_FILENAME, dpi=(72, 72))

    # --- 验证输出结果 ---
    print("-" * 30)
    print(f"文件已保存至: {os.path.abspath(OUTPUT_FILENAME)}")

    # 重新读取验证属性
    with Image.open(OUTPUT_FILENAME) as checked_img:
        print(f"验证分辨率: {checked_img.size} (应为 (912, 1140))")
        print(f"验证模式: {checked_img.mode} (RGB 对应 24-bit)")
        print(f"验证DPI: {checked_img.info.get('dpi')} (应为 (72, 72))")


if __name__ == "__main__":
    main()