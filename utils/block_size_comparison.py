import matplotlib.pyplot as plt
import cv2
import os

# 定义文件映射关系 (Block Size -> 文件名)
# 注意：您上传的文件扩展名混用了 jpg 和 png，这里已做适配
image_map = {
    1: "1.png",
    2: "2.png",
    3: "3.png",
    3: "3.png",
    5: "5.png",
    7: "7.png",
    9: "9.png",
    11: "11.png"
}


def create_comparison_figure():
    # 创建画布：2行3列
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)  # 调整间距

    # 扁平化 axes 数组以便遍历
    axes_flat = axes.flatten()

    sizes = [1, 3, 5, 7, 9, 11]

    print("[INFO] 正在生成对比图...")

    for i, size in enumerate(sizes):
        filename = image_map[size]

        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"[WARN] 找不到文件: {filename}，跳过。")
            continue

        # 读取图像 (读取为灰度)
        img = cv2.imread(filename, 0)

        # 显示图像
        ax = axes_flat[i]
        ax.imshow(img, cmap='gray')

        # 设置标题 (类似论文风格)
        ax.set_title(f"Block Size = {size}", fontsize=14, fontweight='bold', pad=10)

        # 移除坐标轴刻度
        ax.axis('off')

    # 保存结果
    output_filename = "block_size_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] 对比图已保存为: {output_filename}")
    plt.show()


if __name__ == "__main__":
    create_comparison_figure()