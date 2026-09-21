import cv2
import numpy as np
import os
import time
from MvCameraControl_class import *
from ctypes import cdll, byref, sizeof, c_bool, cast, POINTER, c_ubyte

# ====================== 1. 参数配置 ======================

# -- 投影仪参数 --
PROJ_W = 1280  # 投影仪/扩展屏的宽度
PROJ_H = 800  # 投影仪/扩展屏的高度

# -- 主屏幕参数 (用于自动检测失败时的备用方案) --
MANUAL_MAIN_SCREEN_PHYSICAL_W = 2560  # 手动设置: 主屏幕的物理宽度
MANUAL_SCALING_FACTOR = 1.5  # 手动设置: 主屏幕的缩放比例 (150% -> 1.5)

# -- 相机参数 --
EXPOSURE_TIME = 8000.0  # 曝光时间 (微秒)
GAIN = 5.0  # 增益
REVERSE_X = False  # 水平翻转
REVERSE_Y = False  # 垂直翻转

# -- 扫描与保存参数 --
CAPTURE_DELAY_MS = 200  # 投影和拍摄之间的延迟（毫秒）
LEFT_SAVE_PATH = "gray_6.1_view1/left"
RIGHT_SAVE_PATH = "gray_6.1_view1/right"


# ====================== 2. 辅助函数 (源自 camera_control.py) ======================
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
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif pixel_type == PixelType_Gvsp_BayerGB8:
            raw = img_np[:width * height].reshape((height, width))
            return cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2BGR)
        else:
            return None
    except Exception:
        return None


# ====================== 3. 主程序 ======================
if __name__ == "__main__":
    # ===== 新增: 自动检测主屏幕逻辑宽度 =====
    try:
        import win32api

        # GetSystemMetrics(0) 获取主屏幕的逻辑宽度 (已计算缩放)
        MAIN_SCREEN_LOGICAL_W = win32api.GetSystemMetrics(0)
        print(f"[INFO] Successfully detected main screen logical width: {MAIN_SCREEN_LOGICAL_W}px (scaling respected).")
    except ImportError:
        print("[WARNING] 'pywin32' library not found. For automatic screen detection, please run: pip install pywin32")
        MAIN_SCREEN_LOGICAL_W = int(MANUAL_MAIN_SCREEN_PHYSICAL_W / MANUAL_SCALING_FACTOR)
        print(f"[INFO] Falling back to manual calculation. Using logical width: {MAIN_SCREEN_LOGICAL_W}px.")
    except Exception as e:
        print(f"[ERROR] An error occurred during screen detection: {e}")
        MAIN_SCREEN_LOGICAL_W = int(MANUAL_MAIN_SCREEN_PHYSICAL_W / MANUAL_SCALING_FACTOR)
        print(f"[INFO] Falling back to manual calculation. Using logical width: {MAIN_SCREEN_LOGICAL_W}px.")

    # ===== 步骤 1: 生成格雷码图案 =====
    print("[INFO] Generating Gray code patterns...")
    graycode = cv2.structured_light_GrayCodePattern.create(width=PROJ_W, height=PROJ_H)
    ret, patterns = graycode.generate() # graycode.generate() 函数返回的 patterns 是一个元组 (tuple)
    patterns = list(patterns)
    black, white = graycode.getImagesForShadowMasks(np.zeros_like(patterns[0]), np.zeros_like(patterns[0]))
    patterns.append(white)
    patterns.append(black)
    print(f"[INFO] Generated {len(patterns)} patterns for projector resolution ({PROJ_W}, {PROJ_H}).")

    # ===== 步骤 2: 初始化和配置双目相机 =====
    print("[INFO] Initializing cameras...")
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum < 2:
        print(f"[ERROR] Found {deviceList.nDeviceNum} cameras, but 2 are required. Exiting.")
        exit()
    print(f"[INFO] Found {deviceList.nDeviceNum} cameras. Using the first two.")

    cams = []
    for i in range(2):
        stDevInfo = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        cam = MvCamera()
        if cam.MV_CC_CreateHandle(stDevInfo) != 0: continue
        if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0: continue
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
        exit()

    # ===== 步骤 3: 创建保存目录 =====
    os.makedirs(LEFT_SAVE_PATH, exist_ok=True)
    os.makedirs(RIGHT_SAVE_PATH, exist_ok=True)
    print(f"[INFO] Images will be saved to '{LEFT_SAVE_PATH}' and '{RIGHT_SAVE_PATH}'")

    # ===== 步骤 4: 主流程控制 (预览 -> 扫描) =====
    try:
        for i, cam in enumerate(cams):
            cam.MV_CC_StartGrabbing()

        # --- 预览模式 ---
        print("\n[INFO] Entering preview mode. Adjust camera focus and position.")
        print("[INFO] Press 's' to start scanning, or 'q' to quit.")
        while True:
            images_np = [None, None]
            for i, cam in enumerate(cams):
                stFrame = MV_FRAME_OUT()
                if cam.MV_CC_GetImageBuffer(stFrame, 1000) == 0:
                    images_np[i] = process_frame_to_image(stFrame)
                    cam.MV_CC_FreeImageBuffer(stFrame)

            if images_np[0] is not None and images_np[1] is not None:
                preview_img = np.hstack((images_np[0], images_np[1]))
                cv2.imshow("Camera Preview (Left | Right) - Press 's' to start scan", preview_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print("[INFO] Starting scan...")
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                print("[INFO] Quitting...")
                raise SystemExit

        # --- 自动扫描模式 ---

        for i, pattern in enumerate(patterns):
            print(f"[INFO] Projecting and capturing pattern {i + 1}/{len(patterns)}...")
            proj_win_name = 'Projector'
            cv2.namedWindow(proj_win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(proj_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # 使用自动检测或计算出的逻辑宽度来移动窗口
            cv2.moveWindow(proj_win_name, MAIN_SCREEN_LOGICAL_W, 0)
            cv2.imshow(proj_win_name, pattern)
            cv2.waitKey(1)

            time.sleep(CAPTURE_DELAY_MS / 1000.0)

            images_to_save = [None, None]
            capture_success = True
            for cam_idx, cam in enumerate(cams):
                stFrame = MV_FRAME_OUT()
                if cam.MV_CC_GetImageBuffer(stFrame, 1000) == 0:
                    images_to_save[cam_idx] = process_frame_to_image(stFrame)
                    cam.MV_CC_FreeImageBuffer(stFrame)
                else:
                    capture_success = False

            if capture_success and images_to_save[0] is not None and images_to_save[1] is not None:
                left_filename = os.path.join(LEFT_SAVE_PATH, f"{i:02d}.png")
                right_filename = os.path.join(RIGHT_SAVE_PATH, f"{i:02d}.png")
                cv2.imwrite(left_filename, images_to_save[0])
                cv2.imwrite(right_filename, images_to_save[1])
            else:
                print(f"[ERROR] Failed to capture images for pattern {i}.")

            cv2.destroyWindow(proj_win_name)

        print("\n[INFO] Scan complete!")

    except SystemExit:
        pass
    finally:
        # ===== 步骤 5: 停止取流并释放资源 =====
        print("[INFO] Cleaning up resources...")
        for i, cam in enumerate(cams):
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()