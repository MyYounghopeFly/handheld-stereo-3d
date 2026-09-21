import serial
import struct
import time
import math
import os

# EuRoC IMU 数据保存格式表头
EUROC_HEADER = "timestamp,w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n"


def crc16_update(src_bytes, current_crc=0):
    """
    根据手册 C 语言代码翻译的 CRC16 校验算法
    """
    crc = current_crc
    for byte in src_bytes:
        crc = crc ^ (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF  # 保持 16 位
    return crc


class HIPNUC_Reader:
    def __init__(self, port, baudrate=115200):
        # 默认波特率 115200，如果有改动请根据实际情况配置
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        self.buffer = bytearray()

    def read_frame(self):
        """
        从串口读取并解析完整的数据帧
        """
        while True:
            # 每次尝试读取一块数据，减少系统调用开销
            if self.ser.in_waiting > 0:
                self.buffer.extend(self.ser.read(self.ser.in_waiting))
            else:
                self.buffer.extend(self.ser.read(64))  # 阻塞读取

            # 寻找帧头 0x5A 0xA5
            while len(self.buffer) >= 2:
                if self.buffer[0] == 0x5A and self.buffer[1] == 0xA5:
                    break
                else:
                    self.buffer.pop(0)

            # 确保包含帧头(2) + 长度(2) = 4字节
            if len(self.buffer) < 4:
                continue

            # 获取数据域长度 (小端模式)
            payload_len = struct.unpack('<H', self.buffer[2:4])[0]
            # 帧总长 = 帧头(2) + 长度(2) + CRC(2) + Payload(payload_len)
            frame_len = 6 + payload_len

            # 检查缓冲区是否包含完整的帧
            if len(self.buffer) < frame_len:
                continue

            # 提取完整的一帧
            frame = self.buffer[:frame_len]
            self.buffer = self.buffer[frame_len:]  # 截断缓冲区

            # CRC 存放在索引 4, 5 的位置
            recv_crc = struct.unpack('<H', frame[4:6])[0]

            # 提取 Payload (从索引 6 开始)
            payload = frame[6:frame_len]

            # 修复核心点 3：验证 CRC 计算范围包含帧头(0-3) 和 数据域(payload)
            calc_crc = crc16_update(frame[0:4])
            calc_crc = crc16_update(payload, calc_crc)

            if calc_crc != recv_crc:
                print(f"[WARN] CRC 校验失败: Calc 0x{calc_crc:04X}, Recv 0x{recv_crc:04X}")
                continue

            return self.parse_payload(payload)

    def parse_payload(self, payload):
        """
        解析 HI91 数据域 [cite: 353, 356, 357]
        """
        tag = payload[0]
        if tag == 0x91:  # HI91 浮点型数据帧
            if len(payload) != 76:
                return None

            # 使用 struct.unpack 解析二进制数据 (< 表示小端模式)
            # 解析格式: tag(1), pps(2), temp(1), pres(4), time(4), acc_xyz(12), gyr_xyz(12), mag_xyz(12), eul(12), quat(16)
            parsed = struct.unpack('< B H b f I 3f 3f 3f f f f 4f', payload)

            sys_time_ms = parsed[4]
            # G -> m/s^2
            acc_x, acc_y, acc_z = [val * 9.80665 for val in parsed[5:8]]
            # deg/s -> rad/s
            gyr_x, gyr_y, gyr_z = [val * math.pi / 180.0 for val in parsed[8:11]]

            return {
                'sys_time_ms': sys_time_ms,
                'acc': (acc_x, acc_y, acc_z),
                'gyr': (gyr_x, gyr_y, gyr_z)
            }
        return None


if __name__ == "__main__":
    COM_PORT = 'COM5'

    save_dir = "captured_data/imu0"
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, "data.csv")

    imu = HIPNUC_Reader(port=COM_PORT, baudrate=115200)
    print(f"[{COM_PORT}] 串口打开成功，开始接收数据并写入 {csv_file_path} ...")

    with open(csv_file_path, 'w') as f:
        f.write(EUROC_HEADER)

        try:
            count = 0
            while True:
                data = imu.read_frame()
                if data:
                    pc_timestamp_ns = time.time_ns()

                    gx, gy, gz = data['gyr']
                    ax, ay, az = data['acc']

                    line = f"{pc_timestamp_ns},{gx:.6f},{gy:.6f},{gz:.6f},{ax:.6f},{ay:.6f},{az:.6f}\n"
                    f.write(line)

                    count += 1
                    if count % 100 == 0:
                        print(f"已录制 {count} 帧 | 内部系统时间: {data['sys_time_ms']}ms | a_z: {az:.2f} m/s^2")

        except KeyboardInterrupt:
            print("\n停止录制，文件已保存。")