import cv2
import numpy as np
import os
import time
import datetime
import threading
import queue  # 【新增】引入队列模块
import serial
import struct
import math
from ctypes import *
from MvCameraControl_class import *
from CameraParams_header import *

# ====================== 1. 全局与设备参数配置 ======================
COM_PORT = 'COM5'  # ！！！请修改为你的实际 IMU 串口号！！！
IMU_BAUDRATE = 115200

INIT_EXPOSURE_TIME = 9000
INIT_GAIN_RAW = 180

MAX_EXPOSURE_TIME = 100000
MAX_GAIN_RAW = 200

# ================== 目标帧率控制 ==================
TARGET_FPS = 30  # 预览与连续录制的目标帧率
FRAME_INTERVAL_SEC = 1.0 / TARGET_FPS  # 计算每帧的物理时间间隔

# 核心软控制触发配置
MV_TRIGGER_MODE_ON = 1
MV_TRIGGER_SOURCE_SOFTWARE = 7
FRAME_TIMEOUT = 2000
PREVIEW_SCALE = 0.5
WINDOW_NAME = "Stereo-IMU Sync Capture"
ROOT_SAVE_DIR = "SGM_dataset"

EUROC_HEADER = "timestamp,w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n"

is_running = True
is_recording = False

# 【新增】图像保存异步队列，设置最大容量防止内存撑爆
save_queue = queue.Queue(maxsize=1000)


# ====================== 2. IMU 解析底层类 ======================
def crc16_update(src_bytes, current_crc=0):
    crc = current_crc
    for byte in src_bytes:
        crc = crc ^ (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


class HIPNUC_Reader:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        self.buffer = bytearray()

    def read_frame(self):
        while is_running:
            if self.ser.in_waiting > 0:
                self.buffer.extend(self.ser.read(self.ser.in_waiting))
            else:
                self.buffer.extend(self.ser.read(64))

            while len(self.buffer) >= 2:
                if self.buffer[0] == 0x5A and self.buffer[1] == 0xA5:
                    break
                else:
                    self.buffer.pop(0)

            if len(self.buffer) < 4: continue
            payload_len = struct.unpack('<H', self.buffer[2:4])[0]
            frame_len = 6 + payload_len

            if len(self.buffer) < frame_len: continue

            frame = self.buffer[:frame_len]
            self.buffer = self.buffer[frame_len:]

            recv_crc = struct.unpack('<H', frame[4:6])[0]
            payload = frame[6:frame_len]

            calc_crc = crc16_update(frame[0:4])
            calc_crc = crc16_update(payload, calc_crc)

            if calc_crc != recv_crc: continue
            return self.parse_payload(payload)
        return None

    def parse_payload(self, payload):
        tag = payload[0]
        if tag == 0x91:
            if len(payload) != 76: return None
            parsed = struct.unpack('< B H b f I 3f 3f 3f f f f 4f', payload)
            sys_time_ms = parsed[4]
            acc_x, acc_y, acc_z = [val * 9.80665 for val in parsed[5:8]]
            gyr_x, gyr_y, gyr_z = [val * math.pi / 180.0 for val in parsed[8:11]]
            return {'sys_time_ms': sys_time_ms, 'acc': (acc_x, acc_y, acc_z), 'gyr': (gyr_x, gyr_y, gyr_z)}
        return None


# ====================== 3. 后台独立线程 ======================

# A. IMU 线程
def imu_worker(csv_path):
    print(f"[IMU] 正在连接串口 {COM_PORT} ...")
    try:
        imu = HIPNUC_Reader(port=COM_PORT, baudrate=IMU_BAUDRATE)
        print(f"[IMU] 连接成功！后台高频监听中...")
    except Exception as e:
        print(f"[IMU 错误] 无法打开串口 {COM_PORT}: {e}")
        return

    with open(csv_path, 'w') as f:
        f.write(EUROC_HEADER)
        count = 0
        while is_running:
            data = imu.read_frame()
            if data and is_recording:
                pc_timestamp_ns = time.time_ns()
                gx, gy, gz = data['gyr']
                ax, ay, az = data['acc']
                f.write(f"{pc_timestamp_ns},{gx:.6f},{gy:.6f},{gz:.6f},{ax:.6f},{ay:.6f},{az:.6f}\n")
                count += 1
                if count % 500 == 0:
                    print(f"   -> [IMU] 同步录制中，已写入 {count} 帧...")


# 【新增】B. 异步后台存图线程
def save_worker():
    print("[SAVE] 后台高速存图线程已启动...")
    while is_running or not save_queue.empty():
        try:
            # 阻塞等待队列中出现图片，超时1秒继续循环判断退出条件
            task = save_queue.get(timeout=1.0)
            cam0_path, img0, cam1_path, img1 = task

            # 使用更快的压缩级别（1是最快压缩，比3快很多，且不损失画质）
            cv2.imwrite(cam0_path, img0, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(cam1_path, img1, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            save_queue.task_done()
        except queue.Empty:
            pass
    print("[SAVE] 队列已清空，存图线程安全退出。")


# ====================== 4. 视觉与主控制辅助函数 ======================
def process_frame_to_image(stFrame):
    if stFrame.pBufAddr is None or stFrame.stFrameInfo.nFrameLen == 0: return None
    nSize = stFrame.stFrameInfo.nFrameLen
    width, height = stFrame.stFrameInfo.nWidth, stFrame.stFrameInfo.nHeight
    pixel_type = stFrame.stFrameInfo.enPixelType
    pData = (c_ubyte * nSize)()
    cdll.msvcrt.memcpy(byref(pData), stFrame.pBufAddr, nSize)
    data = np.frombuffer(pData, dtype=np.uint8)

    try:
        if pixel_type == PixelType_Gvsp_Mono8:
            return data.reshape((height, width))
        elif pixel_type == PixelType_Gvsp_BGR8_Packed:
            bgr = data.reshape((height, width, 3))
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        elif pixel_type == PixelType_Gvsp_RGB8_Packed:
            rgb = data.reshape((height, width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        elif pixel_type == PixelType_Gvsp_BayerGB8:
            bgr = cv2.cvtColor(data.reshape((height, width)), cv2.COLOR_BAYER_GB2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            return None
    except Exception:
        return None


def create_session_folder():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_path = os.path.join(ROOT_SAVE_DIR, timestamp)
    os.makedirs(os.path.join(session_path, "cam0"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "cam1"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "imu0"), exist_ok=True)
    return session_path, os.path.join(session_path, "cam0"), os.path.join(session_path, "cam1"), os.path.join(
        session_path, "imu0", "data.csv")


def nothing(x): pass


def set_cameras_exposure(cams, exp_us):
    for cam in cams:
        cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        cam.MV_CC_SetFloatValue("ExposureTime", float(exp_us))


def set_cameras_gain(cams, gain_raw):
    val_float = float(gain_raw) / 10.0
    for cam in cams:
        cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
        cam.MV_CC_SetFloatValue("Gain", val_float)


# ====================== 5. 主程序 ======================
if __name__ == "__main__":
    session_dir, cam0_dir, cam1_dir, imu_csv_path = create_session_folder()
    print(f"[SYS] 数据集将保存在: {session_dir}")

    MvCamera.MV_CC_Initialize()
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, deviceList)

    if deviceList.nDeviceNum < 2:
        print(f"[ERROR] 找不到两台相机！(发现 {deviceList.nDeviceNum} 台)")
        exit()

    cams = []
    print(f"[INFO] 正在挂载双目相机并开启软控制 (Software Trigger) 模式...")
    for i in range(2):
        stDevInfo = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        cam = MvCamera()
        if cam.MV_CC_CreateHandle(stDevInfo) != 0: continue
        if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0: continue

        cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)

        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        if ret != 0: print(f"Cam {i} TriggerMode fail: {hex(ret)}")

        ret = cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
        if ret != 0: print(f"Cam {i} TriggerSource fail: {hex(ret)}")

        cam.MV_CC_SetFloatValue("ExposureTime", float(INIT_EXPOSURE_TIME))
        cam.MV_CC_SetFloatValue("Gain", float(INIT_GAIN_RAW / 10.0))

        if cam.MV_CC_StartGrabbing() != 0: continue
        cams.append(cam)
        print(f"[INFO] Cam {i} initialized & live preview ready.")

    if len(cams) < 2:
        exit()

    # 启动后台线程
    imu_th = threading.Thread(target=imu_worker, args=(imu_csv_path,))
    imu_th.start()

    save_th = threading.Thread(target=save_worker)  # 【新增】启动后台存图线程
    save_th.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Exp(us)", WINDOW_NAME, INIT_EXPOSURE_TIME, MAX_EXPOSURE_TIME, nothing)
    cv2.createTrackbar("Gain(0.1dB)", WINDOW_NAME, INIT_GAIN_RAW, MAX_GAIN_RAW, nothing)

    last_exp = INIT_EXPOSURE_TIME
    last_gain = INIT_GAIN_RAW
    img_counter = 0

    print("\n================= 数据集录制面板 =================")
    print(" [ R 键 ]   -> 【开始 / 停止】自动连续拍照 (标定首选)")
    print(" [ S 键 ]   -> 手动拍一张并保存")
    print(" [ Q 键 ]   -> 退出程序")
    print("==================================================")

    display_img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(display_img, "Initializing Cameras...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    last_trigger_time = time.time()
    try:
        while True:
            curr_exp = max(50, cv2.getTrackbarPos("Exp(us)", WINDOW_NAME))
            curr_gain = cv2.getTrackbarPos("Gain(0.1dB)", WINDOW_NAME)
            if curr_exp != last_exp: set_cameras_exposure(cams, curr_exp); last_exp = curr_exp
            if curr_gain != last_gain: set_cameras_gain(cams, curr_gain); last_gain = curr_gain

            cv2.imshow(WINDOW_NAME, display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'): break
            if key == ord('r'):
                is_recording = not is_recording
                print(f"\n>>>> 连续录制状态: {'[ON]' if is_recording else '[OFF]'} <<<<\n")

            current_time = time.time()
            time_to_trigger = (current_time - last_trigger_time >= FRAME_INTERVAL_SEC)

            if time_to_trigger or key == ord('s'):
                last_trigger_time = current_time
                is_save = is_recording or (key == ord('s'))

                trigger_ts_ns = time.time_ns()
                for cam in cams: cam.MV_CC_SetCommandValue("TriggerSoftware")

                frames, success_count = [], 0
                for cam in cams:
                    stFrame = MV_FRAME_OUT()
                    if cam.MV_CC_GetImageBuffer(stFrame, FRAME_TIMEOUT) == 0:
                        frames.append(process_frame_to_image(stFrame))
                        cam.MV_CC_FreeImageBuffer(stFrame)
                        success_count += 1
                    else:
                        frames.append(None)

                if success_count == 2 and frames[0] is not None and frames[1] is not None:
                    if frames[0].shape != frames[1].shape:
                        frames[1] = cv2.resize(frames[1], (frames[0].shape[1], frames[0].shape[0]))

                    # 生成用于 UI 预览的小图
                    concat_img = np.hstack((frames[0], frames[1]))
                    h, w = concat_img.shape[:2]
                    display_img_gray = cv2.resize(concat_img, (int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE)))
                    display_img = cv2.cvtColor(display_img_gray, cv2.COLOR_GRAY2BGR)

                    status_text = "RECORDING" if is_recording else "PREVIEW"
                    if key == ord('s'): status_text = "SAVED SINGLE"
                    color = (0, 0, 255) if is_recording else ((0, 255, 0) if key == ord('s') else (0, 255, 255))

                    cv2.putText(display_img,
                                f"Mode: {status_text} | Saved: {img_counter} | Q Size: {save_queue.qsize()}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(display_img, f"Exp: {curr_exp} | Gain: {curr_gain / 10.0}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # 【核心修改】：不直接存盘，而是扔进后台异步队列
                    if is_save:
                        img_name = f"{trigger_ts_ns}.png"
                        p0 = os.path.join(cam0_dir, img_name)
                        p1 = os.path.join(cam1_dir, img_name)

                        try:
                            # 压入队列，如果写入极度缓慢导致100张的队列塞满，就丢弃这帧防止崩溃
                            save_queue.put_nowait((p0, frames[0], p1, frames[1]))
                            img_counter += 1
                            if not is_recording: print(f"[{status_text}] -> 压入队列 {img_name}")
                        except queue.Full:
                            print(f"[WARN] 硬盘写入速度严重跟不上！队列已满，丢弃帧: {img_name}")

    finally:
        print("\n[SYS] 正在安全停止所有线程与设备...")
        is_running = False

        # 确保队列里的剩余图像写完再彻底关程序
        if not save_queue.empty():
            print(f"[SYS] 等待后台写入剩余的 {save_queue.qsize()} 张图像到硬盘...")
            save_queue.join()

        try:
            imu_th.join()
            save_th.join()  # 等待存图线程收尾
        except NameError:
            pass

        for cam in cams:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        cv2.destroyAllWindows()
        print(f"[SYS] 退出成功！")