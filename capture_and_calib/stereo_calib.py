import cv2
import numpy as np
import os
import glob
import h5py
from MvCameraControl_class import *
from ctypes import cdll, byref, sizeof, cast, POINTER, c_ubyte, c_bool

# ====================== 参数配置 ======================
# --- 相机参数 ---
EXPOSURE_TIME = 8000.0  # 曝光时间（微秒）
GAIN = 8.0  # 增益
REVERSE_X = False  # 水平翻转
REVERSE_Y = False  # 垂直翻转

# --- 标定板参数 ---
BOARD_SIZE = [11, 8]  # 内部角点的数量 (列数, 行数)
SQUARE_SIZE = 25.0  # 棋盘格每个格子实际边长（例如 6.0mm）

# --- 保存路径配置 ---
LEFT_SAVE_PATH = "calib_6_1/left"
RIGHT_SAVE_PATH = "calib_6_1/right"
CALIB_RESULT_PATH = "calib_6_1/stereo_calib_data.h5"


# =======================================================
def save_h5py_file(name, my_dict):
    """将标定结果字典保存为 HDF5 文件"""
    try:
        with h5py.File(name, 'w') as h:
            for k, v in my_dict.items():
                h.create_dataset(k, data=np.array([v]).squeeze())
        print(f"[INFO] Calibration data successfully saved to '{name}'")
    except Exception as e:
        print(f"[ERROR] Failed to save HDF5 file: {e}")


class ChessboardCalibrator(object):
    """棋盘格标定器类"""

    def __init__(self, board_size, square_size):
        self.board_size = tuple(board_size)
        self.square_size = float(square_size)

        # 构造世界坐标点：棋盘格角点在物理世界中的3D坐标。通常将棋盘格定义在Z=0的平面上，所以这些坐标是已知的
        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2) * self.square_size

        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-9)

    def detect_corners_one_img(self, img):
        """在单张图像上检测棋盘格角点"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        ret, corners = cv2.findChessboardCorners(img_gray, self.board_size)
        if ret:
            corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), self.criteria)
            return corners_refined
        return None

    # def generate_map_for_rectification(self, calib_data, size):
    #     """生成用于双目图像校正的映射矩阵"""
    #     M1 = calib_data['M1'].astype(np.float64)
    #     M2 = calib_data['M2'].astype(np.float64)
    #     d1 = calib_data['d1'].astype(np.float64).reshape(1, -1)
    #     d2 = calib_data['d2'].astype(np.float64).reshape(1, -1)
    #     R = calib_data['R'].astype(np.float64)
    #     t = calib_data['t'].astype(np.float64)
    #     R1, R2, P1, P2, Q = cv2.stereoRectify(
    #         cameraMatrix1=M1, cameraMatrix2=M2, distCoeffs1=d1, distCoeffs2=d2, R=R, T=t,
    #         flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, imageSize=size, newImageSize=size
    #     )[0:5]
    #     map_x_l, map_y_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_32FC1)
    #     map_x_r, map_y_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_32FC1)
    #     return {'map_x_l': map_x_l, 'map_y_l': map_y_l, 'map_x_r': map_x_r, 'map_y_r': map_y_r, 'Q': Q}

    def calib_stereo_camera(self, data_dir_l, data_dir_r, size=None):
        """执行双目标定，并可视化角点"""
        img_dir_list_l = sorted(glob.glob(os.path.join(data_dir_l, '*.png')))
        img_dir_list_r = sorted(glob.glob(os.path.join(data_dir_r, '*.png')))
        if not img_dir_list_l or not img_dir_list_r:
            print("[ERROR] No images found in the capture directories. Please capture some images first.")
            return None
        if size is None:
            size = cv2.imread(img_dir_list_l[0], 0).shape[::-1]

        obj_points, img_points_l, img_points_r = [], [], []

        # ===== 为角点生成随机颜色 =====
        num_corners = self.board_size[0] * self.board_size[1]
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_corners)]

        print(f"\n[INFO] Starting calibration with {len(img_dir_list_l)} image pairs...")
        print("[INFO] Press any key to process the next image pair, or 'q' to quit visualization.")

        # ===== 创建可复用的窗口 =====
        cv2.namedWindow('Corner Visualization', cv2.WINDOW_NORMAL)

        for i, (name_l, name_r) in enumerate(zip(img_dir_list_l, img_dir_list_r)):
            # ===== 读取彩色图像用于显示 =====
            img_l = cv2.imread(name_l)
            img_r = cv2.imread(name_r)
            # detect_corners_one_img 内部会自动转为灰度图进行处理
            corners_l = self.detect_corners_one_img(img_l)
            corners_r = self.detect_corners_one_img(img_r)

            if corners_l is not None and corners_r is not None:
                obj_points.append(self.objp)
                img_points_l.append(corners_l)
                img_points_r.append(corners_r)
                print(f"  > Processing pair {i + 1}/{len(img_dir_list_l)}... OK")

                # ===== 可视化角点 =====
                # 遍历所有角点，并在左右图上用相同颜色绘制
                for j in range(num_corners):
                    # 获取整数坐标
                    pt_l = tuple(map(int, corners_l[j].ravel()))
                    pt_r = tuple(map(int, corners_r[j].ravel()))
                    color = colors[j]

                    # 在图上画小圆点
                    cv2.circle(img_l, pt_l, 4, color, -1)
                    cv2.circle(img_r, pt_r, 4, color, -1)

                # 将左/右图像拼接在一起显示
                vis_img = np.hstack((img_l, img_r))
                cv2.imshow('Corner Visualization', vis_img)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quitting visualization.")
                    # 销毁窗口并跳出循环，继续执行标定
                    cv2.destroyWindow('Corner Visualization')
                    # 用户按"q"后还需要把剩下的图片处理完，但不再显示
                    for k in range(i + 1, len(img_dir_list_l)):
                        gray_l = cv2.imread(img_dir_list_l[k], 0)
                        gray_r = cv2.imread(img_dir_list_r[k], 0)
                        corners_l_rem = self.detect_corners_one_img(gray_l)
                        corners_r_rem = self.detect_corners_one_img(gray_r)
                        if corners_l_rem is not None and corners_r_rem is not None:
                            obj_points.append(self.objp)
                            img_points_l.append(corners_l_rem)
                            img_points_r.append(corners_r_rem)
                            print(f"  > Processing pair {k + 1}/{len(img_dir_list_l)}... OK (no viz)")
                        else:
                            print(f"  > Processing pair {k + 1}/{len(img_dir_list_l)}... SKIPPED (no viz)")
                    break  # 跳出外层循环


            else:
                print(f"  > Processing pair {i + 1}/{len(img_dir_list_l)}... SKIPPED")

        try:
            cv2.destroyWindow('Corner Visualization')
        except cv2.error:
            pass  # 窗口不存在则忽略

        if len(obj_points) < 3:
            print("[ERROR] Not enough valid image pairs to perform calibration. At least 3 pairs are needed.")
            return None

        print("[INFO] All image pairs processed. Calibrating...")

        res, M1, d1, M2, d2, R, t, E, F = cv2.stereoCalibrate(
            obj_points, img_points_l, img_points_r, None, None, None, None,
            size, criteria=self.criteria, flags=0
        )
        print(f'[INFO] Stereo calibration done. RMS re-projection error: {res:.5f}')
        calib_data = {'residual': res, 'M1': M1, 'd1': d1, 'M2': M2, 'd2': d2, 'R': R, 't': t, 'E': E, 'F': F}
        print("[INFO] Generating rectification maps...")
        rectify_dict = self.generate_map_for_rectification(calib_data, size)
        calib_data.update(rectify_dict)
        return calib_data


def process_frame_to_image(stFrame):
    """将相机输出的帧数据(stFrame)处理成OpenCV图像(img_np)"""
    if stFrame.pBufAddr is None or stFrame.stFrameInfo.nFrameLen == 0:
        return None
    nSize = stFrame.stFrameInfo.nFrameLen
    data = (c_ubyte * nSize)()
    cdll.msvcrt.memcpy(byref(data), stFrame.pBufAddr, nSize)
    width = stFrame.stFrameInfo.nWidth
    height = stFrame.stFrameInfo.nHeight
    pixel_type = stFrame.stFrameInfo.enPixelType
    img_np = np.frombuffer(data, dtype=np.uint8)
    try:
        if pixel_type == PixelType_Gvsp_Mono8:
            return img_np[:width * height].reshape((height, width))
        elif pixel_type == PixelType_Gvsp_BGR8_Packed:
            return img_np[:width * height * 3].reshape((height, width, 3))
        elif pixel_type == PixelType_Gvsp_RGB8_Packed:
            img_rgb = img_np[:width * height * 3].reshape((height, width, 3))
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # RGB转BGR以适配OpenCV显示
        elif pixel_type == PixelType_Gvsp_BayerGB8:
            raw = img_np[:width * height].reshape((height, width))
            return cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2BGR)
        # ... (其他像素格式转换)
        else:
            print(f"[WARNING] Unsupported pixel type: {pixel_type}. Skipping frame.")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to process frame: {e}")
        return None


if __name__ == "__main__":
    # ===== 1. 初始化与设备检查 =====
    calibrator = ChessboardCalibrator(BOARD_SIZE, SQUARE_SIZE)
    print(f"[INFO] Calibrator initialized for a {BOARD_SIZE[0]}x{BOARD_SIZE[1]} board.")
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum < 2:
        print(f"[ERROR] Found {deviceList.nDeviceNum} cameras, but 2 are required. Exiting.")
        exit()
    print(f"[INFO] Found {deviceList.nDeviceNum} cameras. Using the first two.")

    # ===== 2. 相机初始化与配置  =====
    cams = []
    for i in range(2):
        stDevInfo = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        cam = MvCamera()
        if cam.MV_CC_CreateHandle(stDevInfo) != 0 or cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
            print(f"[ERROR] Camera {i}: Failed to open.")
            continue
        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        cam.MV_CC_SetFloatValue("ExposureTime", EXPOSURE_TIME)
        cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
        cam.MV_CC_SetFloatValue("Gain", GAIN)
        cam.MV_CC_SetBoolValue("ReverseX", c_bool(REVERSE_X))
        cam.MV_CC_SetBoolValue("ReverseY", c_bool(REVERSE_Y))
        cams.append(cam)
        print(f"[INFO] Camera {i} opened and configured successfully.")

    if len(cams) < 2:
        print("[ERROR] Failed to initialize two cameras. Exiting.")
        for cam in cams: cam.MV_CC_DestroyHandle()
        exit()

    # ===== 3. 创建文件夹和计数器  =====
    os.makedirs(LEFT_SAVE_PATH, exist_ok=True)
    os.makedirs(RIGHT_SAVE_PATH, exist_ok=True)
    capture_count = 0
    print(f"[INFO] Images will be saved to '{LEFT_SAVE_PATH}' and '{RIGHT_SAVE_PATH}'")

    # ===== 4. 开始取流与主循环 =====
    try:
        for i, cam in enumerate(cams):
            if cam.MV_CC_StartGrabbing() != 0:
                print(f"[ERROR] Camera {i}: StartGrabbing failed.")
                raise RuntimeError

        print("\n" + "=" * 50)
        print("[INFO] Real-time preview started. Framerate is maximized.")
        print("[INFO] Press 's' to DETECT and SAVE the current frame.")
        print("[INFO] Press 'c' to perform calibration with saved images.")
        print("[INFO] Press 'q' to quit.")
        print("=" * 50 + "\n")

        cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Right Camera", cv2.WINDOW_NORMAL)

        while True:
            images_np = [None, None]
            # 从两个相机获取图像
            for i, cam in enumerate(cams):
                stFrame = MV_FRAME_OUT()
                if cam.MV_CC_GetImageBuffer(stFrame, 1000) == 0:
                    images_np[i] = process_frame_to_image(stFrame)
                else:
                    print(f"[WARNING] Camera {i}: Timeout getting image.")
                if stFrame.pBufAddr is not None:
                    cam.MV_CC_FreeImageBuffer(stFrame)

            if images_np[0] is None or images_np[1] is None:
                continue

            # --- 直接显示原始图像，保证帧率 ---
            cv2.imshow("Left Camera", images_np[0])
            cv2.imshow("Right Camera", images_np[1])

            key = cv2.waitKey(1) & 0xFF

            # --- 按键处理 ---
            if key == ord('s'):
                print("\n[INFO] 's' key pressed. Attempting to detect chessboard...")

                # --- 在此刻进行角点检测 ---
                corners_l = calibrator.detect_corners_one_img(images_np[0])
                corners_r = calibrator.detect_corners_one_img(images_np[1])

                # --- 根据检测结果决定是否保存 ---
                if corners_l is not None and corners_r is not None:
                    left_filename = os.path.join(LEFT_SAVE_PATH, f"{capture_count}.png")
                    right_filename = os.path.join(RIGHT_SAVE_PATH, f"{capture_count}.png")

                    cv2.imwrite(left_filename, images_np[0])
                    cv2.imwrite(right_filename, images_np[1])
                    print(f"[SUCCESS] Chessboard detected in both views. Saved images: {capture_count}.png")
                    capture_count += 1
                else:
                    status_l = "OK" if corners_l is not None else "FAILED"
                    status_r = "OK" if corners_r is not None else "FAILED"
                    print(f"[WARNING] Capture failed. Left: {status_l}, Right: {status_r}. Images not saved.")

            elif key == ord('c'):
                print("\n[INFO] 'c' key pressed. Starting calibration process...")
                calib_results = calibrator.calib_stereo_camera(LEFT_SAVE_PATH, RIGHT_SAVE_PATH)
                if calib_results:
                    save_h5py_file(CALIB_RESULT_PATH, calib_results)
                print("[INFO] Calibration process finished. Quitting...")
                break

            elif key == ord('q'):
                print("[INFO] Quitting...")
                break

    finally:
        # ===== 5. 停止取流并释放资源 =====
        for i, cam in enumerate(cams):
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            print(f"[INFO] Camera {i} released successfully.")
        cv2.destroyAllWindows()