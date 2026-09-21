import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
import h5py
import skimage
import open3d as o3d
from open3d import geometry as o3dg



def numpy_to_o3d(pcd_np):
    valid_ids = (~np.isnan(pcd_np).any(axis=1)) * (~np.isinf(pcd_np).any(axis=1))
    valid_pcd = pcd_np[valid_ids]
    print('there are {} points'.format(valid_pcd.shape[0]))
    tmp = o3dg.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(valid_pcd)
    return tmp


def load_h5py_to_dict(data_dir):
    res = {}
    calib_data_h5 = h5py.File(data_dir, 'r')
    for k, v in calib_data_h5.items():
        res[k] = np.array(v)
    return res


def generate_rectify_data(calib_data, size):
    M1, M2, d1, d2 = calib_data['M1'], calib_data['M2'], calib_data['d1'], calib_data['d2']
    R, t = calib_data['R'], calib_data['t']
    flag = cv2.CALIB_ZERO_DISPARITY

    # cv2.stereoRectify 函数：计算出能让左、右图像共面且行对齐的变换
    # R1, R2: 左、右相机各自需要应用的旋转矩阵，以使其成像平面共面
    # P1, P2: 新的投影矩阵，将三维点投影到校正后的图像平面上
    # Q: 重投影矩阵，理论上可以直接将视差转换为三维坐标，但本脚本未使用它，而是采用了更经典的三角测量法
    R1, R2, P1, P2, Q = cv2.stereoRectify(cameraMatrix1=M1, cameraMatrix2=M2, distCoeffs1=d1, distCoeffs2=d2, R=R, T=t,
                                          flags=flag, alpha=-1, imageSize=size, newImageSize=size)[0:5]

    # 利用 cv2.initUndistortRectifyMap 为左右相机分别生成像素位置的“查找表”（map_x, map_y）
    map_x_l, map_y_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_32FC1)
    map_x_r, map_y_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_32FC1)
    return map_x_l, map_y_l, map_x_r, map_y_r, P1, P2, Q


def rectify(img, map_x, map_y):
    res = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    return res

##--------------------------------------------------------------------


if __name__ == '__main__':
    proj_w, proj_h = 912,1140  # 投影仪的分辨率（宽度*高度）

    # white_thred: 像素值高于此阈值的亮区部分被认为是有效的图案编码
    # black_thred: 在全白和全黑图像对比下，如果一个像素在两幅图中亮度都很低（低于此阈值），则认为该点处于阴影中，不参与解码
    # white_thred, black_thred = 1, 40
    white_thred, black_thred = 0, 100
    img_size = (1280,1024) # 相机图像分辨率

    img_dir_l = './capture_and_calib/scan_data_6.1_view2/left'
    img_dir_r = './capture_and_calib/scan_data_6.1_view2/right'
    calib_data_dir = './capture_and_calib/calib_6_1/stereo_calib_data.h5'
    result_filename = './视角二6.1.ply'
    # result_filename = './test.ply'


    # load camera data
    calib_data = load_h5py_to_dict(calib_data_dir)
    map_x_l, map_y_l, map_x_r, map_y_r, P1, P2, Q = generate_rectify_data(calib_data, size=img_size)

    # decoder
    graycode = cv2.structured_light_GrayCodePattern.create(width=proj_w, height=proj_h)
    graycode.setWhiteThreshold(white_thred) # 过滤掉亮度过低的像素
    graycode.setBlackThreshold(black_thred) # black_thred 应该设置得比阴影区的典型灰度值稍高一些，但要远低于被正常照亮的区域的最低亮度
    num_required_imgs = graycode.getNumberOfPatternImages()
    total_imgs = num_required_imgs + 2  # Graycode序列 + 全白 + 全黑
    print(f'需要读取的图片总数: {total_imgs} (索引 0 到 {total_imgs - 1})')

    def try_read_image(base_dir, idx):
        # 尝试列表：优先尝试拍摄代码使用的 "00.png" 格式，其次尝试 "0.png" 格式
        potential_filenames = [f"{idx:02d}.png", f"{idx}.png"]

        for fname in potential_filenames:
            path = os.path.join(base_dir, fname)
            if os.path.exists(path):
                img = cv2.imread(path, 0)
                if img is not None:
                    return img, path  # 返回成功读取的图片和路径
                else:
                    print(f"[警告] 文件存在但无法读取 (可能是损坏): {path}")

        # 如果循环结束还没返回，说明所有格式都尝试失败
        return None, None

    # load data and rectify:
    rect_list_l, rect_list_r = [], []

    for i in range(total_imgs):
        # 尝试读取左图
        img_l, path_l = try_read_image(img_dir_l, i)
        if img_l is None:
            raise FileNotFoundError(
                f"[致命错误] 无法找到或读取左相机第 {i} 张图片。请检查路径: {img_dir_l} 下是否有 {i:02d}.png 或 {i}.png")

        # 尝试读取右图
        img_r, path_r = try_read_image(img_dir_r, i)
        if img_r is None:
            raise FileNotFoundError(f"[致命错误] 无法找到或读取右相机第 {i} 张图片。请检查路径: {img_dir_r}")

        # 每次读取一对图像，并调用 rectify 函数进行校正
        l_rect, r_rect = rectify(img_l, map_x_l, map_y_l), rectify(img_r, map_x_r, map_y_r)

        # 可视化部分
        # vis_img = np.concatenate([l_rect, r_rect], axis=1)
        # vis_img_small = skimage.transform.rescale(vis_img, 0.5, channel_axis=None)
        # cv2.imshow('rectified data', vis_img_small)
        # cv2.waitKey(1)

        rect_list_l.append(l_rect)
        rect_list_r.append(r_rect)

    # cv2.destroyAllWindows()

    # --- 循环播放预览，等待用户确认 ---
    if len(rect_list_l) > 0:
        print("\n-------------------------------------------------")
        print("所有图像读取完毕。正在循环预览校正效果...")
        print(">>> 检查重点：左右图像的水平特征是否对齐（行对齐）。")
        print("-------------------------------------------------")

        preview_idx = 0
        preview_win_name = 'Rectified Sequence Preview (Loop)'
        num_frames = len(rect_list_l)

        while True:
            # 1. 获取当前帧
            l_img = rect_list_l[preview_idx]
            r_img = rect_list_r[preview_idx]

            # 2. 拼接图像
            concat_img = np.concatenate([l_img, r_img], axis=1)

            # 3. 转换并缩放 (用于显示)
            # 这里的 fx=0.5, fy=0.5 表示显示大小为原图的一半
            vis_img = cv2.resize(concat_img, None, fx=0.5, fy=0.5)

            # 将灰度图转为 BGR，以便画彩色的辅助线
            vis_img_color = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

            # 绘制几条绿色的水平辅助线
            # 这可以帮你直观地判断 立体校正(Rectification) 是否成功
            h, w, _ = vis_img_color.shape
            for y in range(0, h, 100):  # 每隔 100 像素画一条线
                cv2.line(vis_img_color, (0, y), (w, y), (0, 255, 0), 1)

            # 4. 显示
            cv2.imshow(preview_win_name, vis_img_color)

            # 5. 等待按键 (这里设置 100ms，相当于 10fps 的播放速度)
            key = cv2.waitKey(100)

            # 6. 退出条件检测
            # 条件A: 按下 ESC 键 (ASCII 27)
            if key == 27:
                print("用户按下了 ESC，停止预览，继续执行...")
                break

            # 条件B: 点击窗口关闭按钮 (Window Property < 1 表示窗口已销毁)
            try:
                if cv2.getWindowProperty(preview_win_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("窗口被关闭，继续执行...")
                    break
            except:
                pass  # 忽略检测错误

            # 7. 更新索引，实现循环播放
            preview_idx = (preview_idx + 1) % num_frames

        cv2.destroyAllWindows()
    else:
        print("[错误] 没有读取到任何图像，跳过预览。")

    # -------------------------------------------------

    # 用于解码的格雷码图案图像
    pattern_list = np.array([rect_list_l[:-2], rect_list_r[:-2]])  # [:-2]:表示“获取列表中从第一个元素开始，直到但不包括最后两个元素的所有内容”
    white_list = np.array([rect_list_l[-2], rect_list_r[-2]]) # 准备全白图像
    black_list = np.array([rect_list_l[-1], rect_list_r[-1]]) # 准备全黑图像

    # 解码获取视差图
    ret, disp_l = graycode.decode(pattern_list, np.zeros_like(pattern_list[0]), black_list, white_list)

    plt.imshow(disp_l)
    plt.title('disparity map')
    plt.show()

    # 从视差图计算三维点云
    cam_pts_l, cam_pts_r = [], []
    for i in range(disp_l.shape[0]):
        for j in range(disp_l.shape[1]):
            if disp_l[i, j] != 0:
                cam_pts_l.append([j, i])
                cam_pts_r.append([j + disp_l[i, j], i])
    cam_pts_l, cam_pts_r = np.array(cam_pts_l)[:, np.newaxis, :], np.array(cam_pts_r)[:, np.newaxis, :]
    pts4D = cv2.triangulatePoints(P1, P2, np.float32(cam_pts_l), np.float32(cam_pts_r)).T
    pts3D = pts4D[:, :3] / pts4D[:, -1:]

    # 创建 Open3D 点云对象并进行处理
    tmp = numpy_to_o3d(pts3D)
    cl, ind = tmp.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    tmp = tmp.select_by_index(ind)
    print(f"正在保存点云到: {result_filename}")
    o3d.io.write_point_cloud(result_filename, tmp)
    print("保存完成。")

    # --------------------------------------------------------------------
    # 1. 创建代表“左相机”的坐标系模型
    # 它位于世界坐标系原点(0,0,0)
    # frame_size可以根据你的点云大小调整，代表坐标轴的长度
    frame_size = 30  # 假设点云单位是毫米(mm)，创建一个3厘米的坐标轴
    cam_l_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=[0, 0, 0])

    # 2. 创建代表“右相机”的坐标系模型
    # 首先，从标定数据中获取旋转矩阵 R 和平移向量 t
    R = calib_data['R']
    t = calib_data['t']  # t 的单位应与点云单位一致 (例如 mm)

    # 其次，构建一个4x4的齐次变换矩阵
    cam_r_transform = np.eye(4)  # 创建一个4x4的单位矩阵
    cam_r_transform[:3, :3] = R  # 将旋转矩阵R填入左上角3x3
    cam_r_transform[:3, 3] = t   # 将平移向量t填入右上角3x1

    # 最后，创建一个新的坐标系模型，并使用transform()方法将其移动到右相机的位置
    cam_r_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=[0, 0, 0])
    cam_r_frame.transform(cam_r_transform)

    # 3. 将点云、左相机、右相机坐标系一起显示
    print("正在打开3D查看器，显示点云和相机位姿...")
    print("红色=X轴, 绿色=Y轴, 蓝色=Z轴。按 'q' 键关闭窗口。")
    o3d.visualization.draw_geometries(
        [tmp, cam_l_frame, cam_r_frame],  # 将所有要显示的对象放入一个列表
        window_name="点云与相机位姿预览"
    )
