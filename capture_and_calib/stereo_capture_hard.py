import cv2
import numpy as np
import os
import time
import sys
import datetime  # 新增：用于时间戳生成
from ctypes import *
from MvCameraControl_class import *

# ==============================================================================
# 1. 全局参数配置
# ==============================================================================
INIT_EXPOSURE_TIME = 20000.0  # 初始曝光时间 20ms
INIT_GAIN_VALUE = 10.0  # 初始增益

# --- SDK 枚举值 ---
MV_TRIGGER_MODE_ON = 1
MV_TRIGGER_SOURCE_SOFTWARE = 7
MV_TRIGGER_SOURCE_LINE2 = 2

# GPIO 选择
MV_LINE_SELECTOR_LINE2 = 2
MV_LINE_MODE_INPUT = 0
MV_LINE_MODE_STROBE = 8
MV_TRIGGER_EDGE_RISING = 0


# ==============================================================================
# 2. 核心配置函数 (保持不变)
# ==============================================================================
def configure_master_slave(cams):
    """
    配置双目相机为主从模式
    必须物理连接 Pin 5 (Line 2) <--> Pin 5 (Line 2)
    """
    if len(cams) < 2:
        print("[ERROR] 相机数量不足！")
        sys.exit(1)

    # --- A. 配置 Master (左相机) ---
    cam_L = cams[0]
    print(f"\n[CONFIG] 配置 Master (左相机)...")
    cam_L.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
    cam_L.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
    cam_L.MV_CC_SetEnumValue("LineSelector", MV_LINE_SELECTOR_LINE2)
    ret = cam_L.MV_CC_SetEnumValue("LineMode", MV_LINE_MODE_STROBE)
    if ret != 0:
        print("[WARN] Master Line 2 不支持 Strobe，可能固件版本差异")
    cam_L.MV_CC_SetBoolValue("StrobeEnable", True)
    cam_L.MV_CC_SetIntValue("StrobeLineDuration", 500)

    # --- B. 配置 Slave (右相机) ---
    cam_R = cams[1]
    print(f"[CONFIG] 配置 Slave (右相机)...")
    cam_R.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
    ret = cam_R.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE2)
    if ret != 0:
        print(f"[ERROR] Slave 设置 Line2 触发源失败: {hex(ret)}")
    cam_R.MV_CC_SetEnumValue("TriggerActivation", MV_TRIGGER_EDGE_RISING)
    cam_R.MV_CC_SetEnumValue("LineSelector", MV_LINE_SELECTOR_LINE2)
    cam_R.MV_CC_SetEnumValue("LineMode", MV_LINE_MODE_INPUT)

    print("[CONFIG] 主从配置完成。\n")


# ==============================================================================
# 3. 辅助转换函数
# ==============================================================================
def process_frame_to_image(stFrame):
    if stFrame.pBufAddr is None or stFrame.stFrameInfo.nFrameLen == 0:
        return None
    nSize = stFrame.stFrameInfo.nFrameLen
    data = (c_ubyte * nSize)()
    cdll.msvcrt.memcpy(byref(data), stFrame.pBufAddr, nSize)
    img_np = np.frombuffer(data, dtype=np.uint8)

    h = stFrame.stFrameInfo.nHeight
    w = stFrame.stFrameInfo.nWidth

    if stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
        return img_np[:w * h].reshape((h, w))
    elif stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed:
        return img_np[:w * h * 3].reshape((h, w, 3))
    elif stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_BayerGB8:
        raw = img_np[:w * h].reshape((h, w))
        return cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2BGR)
    return None


# ==============================================================================
# 4. 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # --- 1. 创建保存目录结构 ---
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join("captured_data", timestamp_str)
    left_save_dir = os.path.join(base_save_dir, "left")
    right_save_dir = os.path.join(base_save_dir, "right")

    try:
        os.makedirs(left_save_dir, exist_ok=True)
        os.makedirs(right_save_dir, exist_ok=True)
        print(f"[INFO] 图片保存路径: {base_save_dir}")
    except Exception as e:
        print(f"[ERROR] 无法创建目录: {e}")
        sys.exit()

    save_count = 0  # 图片序号计数器

    # --- 2. 初始化相机 ---
    deviceList = MV_CC_DEVICE_INFO_LIST()
    MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)

    if deviceList.nDeviceNum < 2:
        print("未检测到双目相机！")
        sys.exit()

    cams = []
    for i in range(2):
        stDevInfo = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        cam = MvCamera()
        cam.MV_CC_CreateHandle(stDevInfo)
        cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)

        # 统一初始设置
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # 关闭自动曝光
        cam.MV_CC_SetFloatValue("ExposureTime", INIT_EXPOSURE_TIME)
        cam.MV_CC_SetFloatValue("Gain", INIT_GAIN_VALUE)
        cam.MV_CC_SetEnumValue("GainAuto", 0)  # 关闭自动增益

        cams.append(cam)

    # === 执行主从配置 ===
    configure_master_slave(cams)

    # --- 3. 创建窗口并绑定滑条 ---
    win_name = "Stereo View (Sync)"
    cv2.namedWindow(win_name)


    # 定义回调函数 (闭包，可直接访问 cams)
    def update_exposure(val):
        # 曝光限制最小 100us，最大设为 100000us (根据需要调整)
        exp_val = max(100.0, float(val))
        for i, cam in enumerate(cams):
            ret = cam.MV_CC_SetFloatValue("ExposureTime", exp_val)
            if ret != 0:
                print(f"[WARN] Cam{i} 设置曝光失败: {hex(ret)}")
        # print(f"Set Exposure: {exp_val}")


    def update_gain(val):
        gain_val = float(val)
        for i, cam in enumerate(cams):
            ret = cam.MV_CC_SetFloatValue("Gain", gain_val)
            if ret != 0:
                print(f"[WARN] Cam{i} 设置增益失败: {hex(ret)}")
        # print(f"Set Gain: {gain_val}")


    # 创建滑条
    # 曝光范围: 100 - 100000 us
    cv2.createTrackbar("Exp(us)", win_name, int(INIT_EXPOSURE_TIME), 100000, update_exposure)
    # 增益范围: 0 - 20 dB
    cv2.createTrackbar("Gain(dB)", win_name, int(INIT_GAIN_VALUE), 20, update_gain)

    # --- 4. 开始取流 ---
    for cam in cams:
        cam.MV_CC_StartGrabbing()

    print(">>> 系统运行中: 按 's' 保存, 按 'q' 退出")
    print(">>> 拖动滑条可实时调整双目参数")

    try:
        while True:
            # === [修改点 1]：记录触发指令发出前的时间 ===
            t_start = time.perf_counter()

            # [核心] PC 发送软触发 -> Master 曝光 -> Line2 输出 -> Slave 曝光
            ret = cams[0].MV_CC_SetCommandValue("TriggerSoftware")

            frames = [None, None]

            # 获取两路图像
            for i, cam in enumerate(cams):
                stFrame = MV_FRAME_OUT()
                ret = cam.MV_CC_GetImageBuffer(stFrame, 100)
                if ret == 0:
                    frames[i] = process_frame_to_image(stFrame)
                    cam.MV_CC_FreeImageBuffer(stFrame)
                else:
                    print(f"Cam[{i}] 超时 (请检查 Line2 接线是否正确)")

            # === [修改点 2]：图像获取完毕，计算延迟 ===
            if frames[0] is not None and frames[1] is not None:
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000.0

                # 在画面上显示延迟数据
                cv2.putText(frames[0], f"Lat: {latency_ms:.2f} ms", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 打印到控制台
            # print(f"[Time] Trigger-to-Image Delay: {latency_ms:.2f} ms")

            # 双目显示与控制
            if frames[0] is not None and frames[1] is not None:
                # 确保尺寸一致后拼接
                if frames[0].shape == frames[1].shape:
                    vis = np.hstack((frames[0], frames[1]))
                    # 缩放显示，避免窗口太大
                    vis = cv2.resize(vis, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow(win_name, vis)

                key = cv2.waitKey(1) & 0xFF

                # 退出
                if key == ord('q'):
                    break

                # 保存
                if key == ord('s'):
                    # 构造文件名: 0.bmp, 1.bmp ...
                    filename = f"{save_count}.bmp"
                    path_l = os.path.join(left_save_dir, filename)
                    path_r = os.path.join(right_save_dir, filename)

                    cv2.imwrite(path_l, frames[0])
                    cv2.imwrite(path_r, frames[1])

                    print(f"[SAVE] Saved Pair {save_count} to {base_save_dir}")
                    save_count += 1

    finally:
        # 清理资源
        for cam in cams:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()