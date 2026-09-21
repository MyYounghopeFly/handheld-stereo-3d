import os
import csv
import cv2
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

# ================= 配置区 =================
# 填写你的数据集文件夹路径
DATASET_DIR = r"SGM_dataset\bottle9.14"
BAG_NAME = os.path.join(DATASET_DIR, "bottle9.14.bag")


# ==========================================

def create_bag():
    if not os.path.exists(DATASET_DIR):
        print(f"[错误] 找不到文件夹: {DATASET_DIR}")
        return

    print(f"正在准备交织打包...")
    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Imu = typestore.types['sensor_msgs/msg/Imu']
    Image = typestore.types['sensor_msgs/msg/Image']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']

    # 初始化所有消息的全局时间线池
    global_timeline = []

    # 1. 提取所有 IMU 数据
    imu_csv = os.path.join(DATASET_DIR, "imu0", "data.csv")
    if os.path.exists(imu_csv):
        with open(imu_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                ts_ns = int(row[0])
                global_timeline.append((ts_ns, 'imu', row))

    # 2. 提取所有 Cam0 数据
    cam0_dir = os.path.join(DATASET_DIR, "cam0")
    if os.path.exists(cam0_dir):
        for fname in os.listdir(cam0_dir):
            if fname.endswith('.png'):
                ts_ns = int(fname.split('.')[0])
                global_timeline.append((ts_ns, 'cam0', os.path.join(cam0_dir, fname)))

    # 3. 提取所有 Cam1 数据
    cam1_dir = os.path.join(DATASET_DIR, "cam1")
    if os.path.exists(cam1_dir):
        for fname in os.listdir(cam1_dir):
            if fname.endswith('.png'):
                ts_ns = int(fname.split('.')[0])
                global_timeline.append((ts_ns, 'cam1', os.path.join(cam1_dir, fname)))

    # 【核心修复】：对全局时间线进行严格的时间戳升序排序 (Interleaving)
    print(f"共提取到 {len(global_timeline)} 条数据，正在进行时间线跨界排序...")
    global_timeline.sort(key=lambda x: x[0])

    # 开始写入 Bag
    with Writer(BAG_NAME) as bag:
        topic_imu = bag.add_connection('/imu0', msgtype='sensor_msgs/msg/Imu', typestore=typestore)
        topic_cam0 = bag.add_connection('/cam0/image_raw', msgtype='sensor_msgs/msg/Image', typestore=typestore)
        topic_cam1 = bag.add_connection('/cam1/image_raw', msgtype='sensor_msgs/msg/Image', typestore=typestore)


        cam0_seq = 0
        cam1_seq = 0
        imu_seq = 0

        print("开始严格按真实物理时间轴写入 .bag 文件...")
        for idx, msg in enumerate(global_timeline):
            ts_ns, msg_type, data = msg
            sec = ts_ns // 1_000_000_000
            nanosec = ts_ns % 1_000_000_000
            msg_time = Time(sec=sec, nanosec=nanosec)

            if msg_type == 'imu':
                header = Header(stamp=msg_time, frame_id='imu', seq=imu_seq)
                # 提取角速度和线加速度
                gx, gy, gz, ax, ay, az = map(float, data[1:7])
                imu_msg = Imu(
                    header=header,
                    orientation=typestore.types['geometry_msgs/msg/Quaternion'](0.0, 0.0, 0.0, 1.0),
                    orientation_covariance=np.zeros(9, dtype=np.float64),
                    angular_velocity=Vector3(gx, gy, gz),
                    angular_velocity_covariance=np.zeros(9, dtype=np.float64),
                    linear_acceleration=Vector3(ax, ay, az),
                    linear_acceleration_covariance=np.zeros(9, dtype=np.float64)
                )
                bag.write(topic_imu, ts_ns, typestore.serialize_ros1(imu_msg, 'sensor_msgs/msg/Imu'))
                imu_seq += 1

            elif msg_type in ['cam0', 'cam1']:
                img_path = data
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue

                header = Header(stamp=msg_time, frame_id=msg_type, seq=(cam0_seq if msg_type == 'cam0' else cam1_seq))
                height, width = img.shape

                img_msg = Image(
                    header=header,
                    height=height,
                    width=width,
                    encoding='mono8',
                    is_bigendian=0,
                    step=width,
                    data=img.ravel()
                )

                if msg_type == 'cam0':
                    bag.write(topic_cam0, ts_ns, typestore.serialize_ros1(img_msg, 'sensor_msgs/msg/Image'))
                    cam0_seq += 1
                else:
                    bag.write(topic_cam1, ts_ns, typestore.serialize_ros1(img_msg, 'sensor_msgs/msg/Image'))
                    cam1_seq += 1

            if idx % 1000 == 0 and idx > 0:
                print(f"  已按时间线写入 {idx} / {len(global_timeline)} 条数据...")

    print(f"[完成] 交织打包结束！高频无损文件已保存至: {BAG_NAME}")


if __name__ == "__main__":
    create_bag()