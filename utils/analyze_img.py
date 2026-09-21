import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


class ImageInspectorTool:
    def __init__(self, image_path):
        """初始化工具"""
        self.image_path = image_path
        self.img_original = None  # 原始图像，保持不变
        self.img_display = None  # 用于显示的图像，会在上面绘制点和线
        self.img_gray = None  # 灰度版本的图像，用于快速查询灰度值
        self.window_name = 'Image Inspector & Distance Tool'
        # 用于存储用户点击的点 [(x1, y1), (x2, y2)]
        self.points = []

    def load_image(self):
        """加载图片并创建灰度副本"""
        self.img_original = cv2.imread(self.image_path)
        if self.img_original is None:
            print(f"错误：无法加载图片，请检查文件路径或文件格式：{self.image_path}")
            return False

        # 创建一个用于显示的副本
        self.img_display = self.img_original.copy()

        # 创建一个灰度版本的图像用于查询像素值
        if len(self.img_original.shape) == 3:
            self.img_gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        else:
            self.img_gray = self.img_original.copy()

        return True

    def mouse_event_handler(self, event, x, y, flags, param):
        """鼠标事件的核心处理函数"""
        # --- 功能1: 实时显示灰度值 (通过鼠标移动触发) ---
        if event == cv2.EVENT_MOUSEMOVE:
            # 制作一个临时图像副本，以避免永久性地在图上留下上一次的文本
            temp_display = self.img_display.copy()

            # 如果正在选择第二个点，画一条"橡皮筋"线
            if len(self.points) == 1:
                cv2.line(temp_display, self.points[0], (x, y), (255, 0, 0), 1)

            # 准备灰度值文本
            gray_value = self.img_gray[y, x]
            text = f"XY:({x},{y})  Gray:{gray_value}"

            # 为了确保文本清晰可见，先画一个黑色背景
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(temp_display, (0, 0), (text_width + 15, text_height + 15), (0, 0, 0), -1)
            # 再画上文本
            cv2.putText(temp_display, text, (10, text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # 显示这个带有所有临时信息的图像
            cv2.imshow(self.window_name, temp_display)

        # --- 功能2: 测量像素距离 (通过鼠标左键点击触发) ---
        elif event == cv2.EVENT_LBUTTONDOWN:
            # 如果已经选了2个点, 这次点击将开始新的测量
            if len(self.points) >= 2:
                self.points = []
                self.img_display = self.img_original.copy()  # 重置显示图像

            # 添加新的点
            self.points.append((x, y))
            # 在新点上画一个永久性的圆作为标记
            cv2.circle(self.img_display, (x, y), 5, (0, 255, 0), -1)

            # 如果刚好选了两个点
            if len(self.points) == 2:
                p1 = self.points[0]
                p2 = self.points[1]
                # 在两点之间画一条永久性的线
                cv2.line(self.img_display, p1, p2, (0, 0, 255), 2)

                # 计算并显示距离
                distance = np.linalg.norm(np.array(p1) - np.array(p2))
                dist_text = f"Distance: {distance:.2f} pixels"
                print(f"Point 1: {p1}, Point 2: {p2}, Distance: {distance:.2f} pixels")

                # 在图像左下角显示距离
                cv2.putText(self.img_display, dist_text, (10, self.img_display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 更新显示，这次是永久性的绘制
            cv2.imshow(self.window_name, self.img_display)

    def run(self):
        """运行主程序"""
        if not self.load_image():
            return

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_event_handler)

        print("--- 操作指南 ---")
        print("1. 移动鼠标: 实时查看左上角的坐标和灰度值。")
        print("2. 单击左键: 选择一个点进行距离测量。")
        print("3. 再次单击: 选择第二个点，自动计算并显示距离。")
        print("4. 若要重新测量，只需再次单击即可开始。")
        print("5. 按 'q' 键或 'ESC' 键退出。")

        cv2.imshow(self.window_name, self.img_display)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()
        print("程序已退出。")


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    print("请在弹出的窗口中选择一个图片文件...")
    file_path = filedialog.askopenfilename(
        title="请选择图片文件",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
    )

    if file_path:
        tool = ImageInspectorTool(file_path)
        tool.run()
    else:
        print("没有选择文件，程序退出。")