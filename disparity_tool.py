import cv2
import numpy as np
import matplotlib.pyplot as plt


def estimate_disparity_range(left_img_path, right_img_path):
    print(f"[INFO] 正在分析图像...")

    # 1. 读取图像 (必须是校正后的灰度图)
    imgL = cv2.imread(left_img_path, 0)
    imgR = cv2.imread(right_img_path, 0)

    if imgL is None or imgR is None:
        print("错误: 无法读取图像，请检查路径。")
        return

    # 2. 使用 ORB 特征检测器 (速度快，对纹理敏感)
    # nfeatures=2000 保证提取足够多的点
    orb = cv2.ORB_create(nfeatures=2000)

    # 3. 检测关键点和描述符
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)

    if des1 is None or des2 is None:
        print("错误: 无法提取特征点，图像可能太暗或无纹理。")
        return

    # 4. 特征匹配 (汉明距离)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 按距离排序，取最好的匹配点
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. 提取视差并过滤异常值
    disparities = []
    valid_matches = []

    print(f"[INFO] 找到原始匹配点: {len(matches)} 个")

    for m in matches:
        ptL = kp1[m.queryIdx].pt
        ptR = kp2[m.trainIdx].pt

        disp = ptL[0] - ptR[0]  # 视差 = x_left - x_right
        dy = abs(ptL[1] - ptR[1])  # 垂直差异 (极线误差)

        # 过滤逻辑：
        # 1. 极线误差必须很小 (校正后的图 y 应该对齐)
        # 2. 这里的阈值设为 3 像素，宽容一点
        if dy < 2.0:
            disparities.append(disp)
            valid_matches.append(m)

    if not disparities:
        print("错误: 没有找到满足极线约束的匹配点。请检查图片是否已做极线校正。")
        return

    # 6. 统计分析 (使用百分位数剔除离群噪点)
    disparities = np.array(disparities)

    # 取 5% 和 95% 分位数作为稳健的最小/最大值，防止个别错误匹配干扰
    min_val = np.percentile(disparities, 5)
    max_val = np.percentile(disparities, 95)

    print(f"\n{'=' * 40}")
    print(f" >>> 视差统计结果 (基于 {len(disparities)} 个有效特征点) <<<")
    print(f" 视差最小值 (Raw Min): {np.min(disparities):.2f}")
    print(f" 视差最大值 (Raw Max): {np.max(disparities):.2f}")
    print(f" 稳健区间 [5% - 95%]: [{min_val:.2f}, {max_val:.2f}]")
    print(f"{'=' * 40}")

    # 7. 计算 SGM 建议参数
    # minDisparity 向下取整到 16 的倍数
    suggested_min_disp = int(np.floor(min_val / 16.0) * 16) - 16  # 多留 16px 余量

    # numDisparities
    range_width = max_val - suggested_min_disp
    suggested_num_disp = int(np.ceil(range_width / 16.0) * 16) + 32  # 多留 32px 余量

    print(f"\n[建议 SGM 参数]")
    print(f" minDisparity   = {suggested_min_disp}")
    print(f" numDisparities = {suggested_num_disp}")
    print(f" (搜索范围: {suggested_min_disp} 到 {suggested_min_disp + suggested_num_disp})")

    # 8. 可视化视差分布
    plt.figure(figsize=(10, 5))
    plt.hist(disparities, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(min_val, color='r', linestyle='dashed', linewidth=1, label=f'Min (5%): {min_val:.1f}')
    plt.axvline(max_val, color='r', linestyle='dashed', linewidth=1, label=f'Max (95%): {max_val:.1f}')
    plt.title('Disparity Distribution')
    plt.xlabel('Disparity (pixels)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 可视化匹配结果
    img_match = cv2.drawMatches(imgL, kp1, imgR, kp2, valid_matches[:50], None, flags=2)
    cv2.imshow("Top 50 Matches (Verified)", img_match)
    print("\n按任意键关闭匹配图...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 确保这里使用的是【校正后】的灰度图
    IMG_L = "rectified_left.png"
    IMG_R = "rectified_right.png"

    estimate_disparity_range(IMG_L, IMG_R)
