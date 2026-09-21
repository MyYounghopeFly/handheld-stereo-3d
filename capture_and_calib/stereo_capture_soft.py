import cv2
import numpy as np
import os
import time
import datetime
from ctypes import *
from MvCameraControl_class import *
from CameraParams_header import *

# ====================== 1. 参数配置 ======================

# -- 初始参数 --
INIT_EXPOSURE_TIME = 20000  # 默认曝光 20ms
INIT_GAIN_RAW = 100  # 默认增益 10.0dB

# -- 滑条范围 --
MAX_EXPOSURE_TIME = 100000
MAX_GAIN_RAW = 200

# -- 触发配置 (关键修改) --
MV_TRIGGER_MODE_ON = 1
MV_TRIGGER_SOURCE_SOFTWARE = 7

# -- 其他 --
FRAME_TIMEOUT = 2000  # 触发模式下超时时间稍微设长一点
PREVIEW_SCALE = 0.5
WINDOW_NAME = "Stereo True Soft Trigger"
ROOT_SAVE_DIR = "captured_data_soft_trig"


# ====================== 2. 辅助函数 ======================

def process_frame_to_image(stFrame):
    """SDK帧转OpenCV图像"""
    if stFrame.pBufAddr is None or stFrame.stFrameInfo.nFrameLen == 0:
        return None

    nSize = stFrame.stFrameInfo.nFrameLen
    width = stFrame.stFrameInfo.nWidth
    height = stFrame.stFrameInfo.nHeight
    pixel_type = stFrame.stFrameInfo.enPixelType

    pData = (c_ubyte * nSize)()
    cdll.msvcrt.memcpy(byref(pData), stFrame.pBufAddr, nSize)
    data = np.frombuffer(pData, dtype=np.uint8)

    try:
        if pixel_type == PixelType_Gvsp_Mono8:
            image = data.reshape((height, width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif pixel_type == PixelType_Gvsp_BGR8_Packed:
            return data.reshape((height, width, 3))
        elif pixel_type == PixelType_Gvsp_RGB8_Packed:
            return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)
        elif pixel_type == PixelType_Gvsp_BayerGB8:
            return cv2.cvtColor(data.reshape((height, width)), cv2.COLOR_BAYER_GB2BGR)
        else:
            return None
    except Exception:
        return None


def create_session_folder():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_path = os.path.join(ROOT_SAVE_DIR, timestamp)
    os.makedirs(os.path.join(session_path, "left"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "right"), exist_ok=True)
    return session_path, os.path.join(session_path, "left"), os.path.join(session_path, "right")


def nothing(x):
    pass


def set_cameras_exposure(cams, exp_us):
    for cam in cams:
        # 触发模式下通常需要先停流或直接设，海康SDK一般允许运行中改，但需注意ExposureAuto必须为OFF
        cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        cam.MV_CC_SetFloatValue("ExposureTime", float(exp_us))


def set_cameras_gain(cams, gain_raw):
    val_float = float(gain_raw) / 10.0
    for cam in cams:
        cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
        cam.MV_CC_SetFloatValue("Gain", val_float)


# ====================== 3. 主程序 ======================

if __name__ == "__main__":
    MvCamera.MV_CC_Initialize()

    # 1. 枚举设备
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, deviceList)
    if deviceList.nDeviceNum < 2:
        print(f"[ERROR] Need at least 2 cameras. Found {deviceList.nDeviceNum}.")
        exit()

    cams = []
    print(f"[INFO] Initializing cameras in SOFT TRIGGER mode...")

    # 2. 初始化双目相机
    for i in range(2):
        stDevInfo = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        cam = MvCamera()
        if cam.MV_CC_CreateHandle(stDevInfo) != 0: continue
        if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0: continue

        # === [关键修改] 开启软触发模式 ===
        # 必须先关闭自动触发和自动曝光/增益
        cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)

        # 设置为触发模式 ON
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        if ret != 0: print(f"Cam {i} TriggerMode fail: {hex(ret)}")

        # 设置触发源为 Software (7)
        ret = cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
        if ret != 0: print(f"Cam {i} TriggerSource fail: {hex(ret)}")

        # 应用初始参数
        cam.MV_CC_SetFloatValue("ExposureTime", float(INIT_EXPOSURE_TIME))
        cam.MV_CC_SetFloatValue("Gain", float(INIT_GAIN_RAW / 10.0))

        if cam.MV_CC_StartGrabbing() != 0: continue
        cams.append(cam)
        print(f"[INFO] Cam {i} initialized & waiting for trigger command.")

    if len(cams) < 2: exit()

    # 3. 创建GUI
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Exp(us)", WINDOW_NAME, INIT_EXPOSURE_TIME, MAX_EXPOSURE_TIME, nothing)
    cv2.createTrackbar("Gain(0.1dB)", WINDOW_NAME, INIT_GAIN_RAW, MAX_GAIN_RAW, nothing)

    last_exp = INIT_EXPOSURE_TIME
    last_gain = INIT_GAIN_RAW

    # 建立保存目录
    session_dir, left_dir, right_dir = create_session_folder()
    img_counter = 0

    print("\n=== 操作指南 (真·软触发) ===")
    print(" [空格键] 拍照预览 (计算延迟，不保存)")
    print(" [S 键]   拍照并保存")
    print(" [Q 键]   退出")
    print("==========================")

    # 用于在界面上保留显示的图像（因为不触发时没有新图）
    display_img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(display_img, "Press SPACE to Trigger", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    try:
        while True:
            # --- A. 参数同步 ---
            # 即使不拍照，也检查滑条，以便下次拍照时生效
            curr_exp = cv2.getTrackbarPos("Exp(us)", WINDOW_NAME)
            curr_gain = cv2.getTrackbarPos("Gain(0.1dB)", WINDOW_NAME)
            if curr_exp < 50: curr_exp = 50

            if curr_exp != last_exp:
                set_cameras_exposure(cams, curr_exp)
                last_exp = curr_exp
            if curr_gain != last_gain:
                set_cameras_gain(cams, curr_gain)
                last_gain = curr_gain

            # 显示当前（或上一帧）图像
            cv2.imshow(WINDOW_NAME, display_img)

            # --- B. 等待按键 (这是软触发的核心) ---
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break

            # 只有当按下 's' (保存) 或 ' ' (预览) 时，才触发相机
            elif key == ord('s') or key == ord(' '):
                is_save = (key == ord('s'))

                # 1. 记录开始时间
                t_start = time.perf_counter()

                # 2. 发送软触发命令 (两台相机依次触发，会有微小时间差，这是纯软触发的特性)
                for cam in cams:
                    cam.MV_CC_SetCommandValue("TriggerSoftware")

                # 3. 获取图像
                frames = []
                success_count = 0
                for cam in cams:
                    stFrame = MV_FRAME_OUT()
                    # 超时时间设为 1000ms 以上，保证曝光完成
                    ret = cam.MV_CC_GetImageBuffer(stFrame, FRAME_TIMEOUT)
                    if ret == 0:
                        frames.append(process_frame_to_image(stFrame))
                        cam.MV_CC_FreeImageBuffer(stFrame)
                        success_count += 1
                    else:
                        print(f"[WARN] GetImage Failed/Timeout: {hex(ret)}")
                        frames.append(None)

                # 4. 记录结束时间并计算延迟
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000.0

                # 5. 处理图像
                if success_count == 2 and frames[0] is not None and frames[1] is not None:
                    # 对齐与拼接
                    if frames[0].shape != frames[1].shape:
                        frames[1] = cv2.resize(frames[1], (frames[0].shape[1], frames[0].shape[0]))
                    concat_img = np.hstack((frames[0], frames[1]))

                    # 缩放用于显示
                    h, w = concat_img.shape[:2]
                    display_img = cv2.resize(concat_img, (int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE)))

                    # 绘制延迟信息
                    status_text = "SAVE" if is_save else "PREVIEW"
                    color = (0, 255, 0) if is_save else (0, 255, 255)

                    cv2.putText(display_img, f"Mode: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(display_img, f"Latency: {latency_ms:.2f} ms", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
                    cv2.putText(display_img, f"Exp: {curr_exp} | Gain: {curr_gain / 10.0}", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    print(f"[{status_text}] Latency: {latency_ms:.2f}ms")

                    # 6. 保存逻辑
                    if is_save:
                        name = f"img_{img_counter:04d}.bmp"
                        cv2.imwrite(os.path.join(left_dir, name), frames[0])
                        cv2.imwrite(os.path.join(right_dir, name), frames[1])
                        print(f"       -> Saved {name}")
                        img_counter += 1
                else:
                    print("[ERROR] Capture failed (one or both cameras timed out)")

    finally:
        for cam in cams:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        cv2.destroyAllWindows()