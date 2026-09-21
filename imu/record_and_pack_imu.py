import serial
import struct
import time
import math
import sys
import os
import csv
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

# ====================== 1. 全局配置区 ======================
COM_PORT = 'COM5'  # ！！！请修改为你的实际 IMU 串口号！！！
IMU_BAUDRATE = 115200  # 串口波特率
RECORD_MINUTES = 120  # 目标录制时长 (分钟)

# 临时 CSV 文件和最终生成的 BAG 文件名
OUTPUT_CSV = "static_imu_temp.csv"
OUTPUT_BAG = "static_imu_120min.bag"
TOPIC_NAME = "/imu0"  # imu_utils 默认读取的 Topic

EUROC_HEADER = "timestamp,w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n"


# ====================== 2. HIPNUC 解析底层 ======================
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
        if self.ser.in_waiting > 0:
            self.buffer.extend(self.ser.read(self.ser.in_waiting))
        else:
            self.buffer.extend(self.ser.read(64))

        while len(self.buffer) >= 2:
            if self.buffer[0] == 0x5A and self.buffer[1] == 0xA5:
                break
            else:
                self.buffer.pop(0)

        if len(self.buffer) < 4: return None
        payload_len = struct.unpack('<H', self.buffer[2:4])[0]
        frame_len = 6 + payload_len

        if len(self.buffer) < frame_len: return None

        frame = self.buffer[:frame_len]
        self.buffer = self.buffer[frame_len:]

        recv_crc = struct.unpack('<H', frame[4:6])[0]
        payload = frame[6:frame_len]

        calc_crc = crc16_update(frame[0:4])
        calc_crc = crc16_update(payload, calc_crc)

        if calc_crc != recv_crc: return None

        tag = payload[0]
        if tag == 0x91 and len(payload) == 76:
            parsed = struct.unpack('< B H b f I 3f 3f 3f f f f 4f', payload)
            acc_x, acc_y, acc_z = [val * 9.80665 for val in parsed[5:8]]
            gyr_x, gyr_y, gyr_z = [val * math.pi / 180.0 for val in parsed[8:11]]
            return {'acc': (acc_x, acc_y, acc_z), 'gyr': (gyr_x, gyr_y, gyr_z)}
        return None


# ====================== 3. 后处理：打包成 ROS Bag ======================
def convert_csv_to_bag(csv_file, bag_file):
    print(f"\n==================================================")
    print(f"[BAG PACKING] 正在将数据封装为 ROS1 标准 .bag 文件...")
    if not os.path.exists(csv_file):
        print(f"[ERROR] 找不到临时文件: {csv_file}")
        return

    # 挂载 ROS1 标准数据类型库
    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Imu = typestore.types['sensor_msgs/msg/Imu']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']

    with Writer(bag_file) as bag:
        topic_imu = bag.add_connection(TOPIC_NAME, 'sensor_msgs/msg/Imu', typestore=typestore)
        imu_count = 0

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头

            for row in reader:
                try:
                    ts_ns = int(row[0])
                    gx, gy, gz = float(row[1]), float(row[2]), float(row[3])
                    ax, ay, az = float(row[4]), float(row[5]), float(row[6])

                    sec = ts_ns // 1_000_000_000
                    nanosec = ts_ns % 1_000_000_000
                    msg_time = Time(sec=sec, nanosec=nanosec)
                    header = Header(stamp=msg_time, frame_id='imu', seq=imu_count)

                    imu_msg = Imu(
                        header=header,
                        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                        orientation_covariance=np.zeros(9, dtype=np.float64),
                        angular_velocity=Vector3(x=gx, y=gy, z=gz),
                        angular_velocity_covariance=np.zeros(9, dtype=np.float64),
                        linear_acceleration=Vector3(x=ax, y=ay, z=az),
                        linear_acceleration_covariance=np.zeros(9, dtype=np.float64)
                    )
                    bag.write(topic_imu, ts_ns, typestore.serialize_ros1(imu_msg, 'sensor_msgs/msg/Imu'))
                    imu_count += 1
                except Exception:
                    continue

    print(f"[SUCCESS] 打包完成！成功注入 {imu_count} 帧 IMU 报文。")
    print(f"[SUCCESS] 你的 Bag 文件已生成: {os.path.abspath(bag_file)}")

    # 打包完成后，自动删除临时的 CSV 文件以节省空间
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"[INFO] 临时日志文件已自动清理。")
    print(f"==================================================\n")


# ====================== 4. 核心录制流水线 ======================
def main():
    print(f"==================================================")
    print(f"[INIT] 准备录制 IMU 静态标定数据")
    print(f"       端口: {COM_PORT} | 波特率: {IMU_BAUDRATE}")
    print(f"       目标时长: {RECORD_MINUTES} 分钟")
    print(f"==================================================")

    input(">>> 请将设备静置于坚固台面上，确认无误后按 [Enter] 键开始录制...")

    try:
        imu = HIPNUC_Reader(port=COM_PORT, baudrate=IMU_BAUDRATE)
        print(f"[SUCCESS] 串口连接成功，开始录制！切勿触碰设备...")
    except Exception as e:
        print(f"[ERROR] 无法打开串口: {e}")
        return

    record_seconds = RECORD_MINUTES * 60
    start_time = time.time()
    last_print_time = start_time
    count = 0

    try:
        with open(OUTPUT_CSV, 'w') as f:
            f.write(EUROC_HEADER)

            while True:
                current_time = time.time()
                elapsed = current_time - start_time

                # 检查是否达到录制时间
                if elapsed >= record_seconds:
                    print(f"\n[FINISH] 录制时间达标！共收集 {count} 帧数据。")
                    break

                data = imu.read_frame()
                if data:
                    ts_ns = time.time_ns()
                    gx, gy, gz = data['gyr']
                    ax, ay, az = data['acc']

                    f.write(f"{ts_ns},{gx:.6f},{gy:.6f},{gz:.6f},{ax:.6f},{ay:.6f},{az:.6f}\n")
                    count += 1

                # 每隔 10 秒打印一次进度
                if current_time - last_print_time >= 10:
                    remaining = record_seconds - elapsed
                    m, s = divmod(int(remaining), 60)
                    print(f"   -> [正在记录] 已收集 {count} 帧... 剩余时间: {m}分 {s}秒")
                    last_print_time = current_time

    except KeyboardInterrupt:
        print(f"\n\n[WARN] 检测到手动中断 (Ctrl+C)！")
        print(f"[WARN] 正在保存已录制的 {count} 帧数据...")

    # 无论是否跑满 2 小时，还是被手动中断，最终都会自动执行打包
    if count > 0:
        convert_csv_to_bag(OUTPUT_CSV, OUTPUT_BAG)
    else:
        print("[ERROR] 没有录制到任何数据，已取消打包。")


if __name__ == "__main__":
    main()