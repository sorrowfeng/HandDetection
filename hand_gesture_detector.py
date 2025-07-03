import cv2
import mediapipe as mp
import numpy as np
import zmq
import json
from datetime import datetime
import os
import sys
import signal



# 定义每根手指的三个关节点（CMC → MCP → PIP → DIP → TIP）
FINGER_JOINTS = {
    'thumb': [[0, 1, 2], [1, 2, 3], [2, 3, 4]],       # 拇指
    'index': [[0, 5, 6], [5, 6, 7], [6, 7, 8]],       # 食指
    'middle': [[0, 9, 10], [9, 10, 11], [10, 11, 12]],  # 中指
    'ring': [[0, 13, 14], [13, 14, 15], [14, 15, 16]],  # 无名指
    'pinky': [[0, 17, 18], [17, 18, 19], [18, 19, 20]]  # 小指
}


def resource_path(relative_path):
    """ 获取资源绝对路径，适用于 PyInstaller 打包后的环境 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def show_simple(frame, angles, font_scale=0.6, text_color=(0, 255, 0)):
    """
    最简实时显示函数：在图像上叠加角度文本并显示窗口
    :param frame: 输入图像（BGR格式，已绘制关键点）
    :param angles: 角度字典（如 {'thumb': [...], 'index': [...], ...}）
    :param font_scale: 字体大小
    :param text_color: 文本颜色（BGR格式）
    """
    # 复制图像避免修改原始数据
    img = frame.copy()
    
    # 显示图像（窗口自动适应尺寸）
    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)  # 实时刷新（1ms延迟）



class HandGestureDetector:
    def __init__(self):
        self.running = True
        self.show_window = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        # 初始化平滑滤波参数
        self.smoothing_factor = 0.3  # 平滑系数（0.1~0.5，越小越平滑）
        self.previous_landmarks = None  # 存储上一帧的关键点数据

        # 初始化OpenCV卡尔曼滤波器
        self.kalman_filters = {}
        self.initialize_opencv_kalman_filters()

        # 设置 Mediapipe 自定义模型路径（打包后也能找到）
        mp_path = resource_path('mediapipe')
        os.environ['MEDIAPIPE_HOME'] = mp_path

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # 开启 model_complexity=1 以启用 3D 坐标
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        # ZeroMQ 设置
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")  # 监听端口 5555
        print("[INFO] ZMQ Publisher started on tcp://*:5555")

    def exit_gracefully(self, *args):
        print("[INFO] Exiting gracefully...")
        self.running = False

    def set_show_window(self, show):
        self.show_window = show

    def calculate_angle_2d(self, a, b, c):
        """
        输入三个 2D 点 a(x,y), b(x,y), c(x,y)，计算 ∠abc 的角度（单位为度）
        """
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians))
        return min(angle, 360 - angle)  # 返回最小内角

    def calculate_angle_3d(self, a, b, c):
        """
        输入三个 3D 点 a, b, c，计算 ∠abc 的角度（单位为度）
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        # 单位向量
        ba_norm = ba / np.linalg.norm(ba) if np.linalg.norm(ba) != 0 else ba
        bc_norm = bc / np.linalg.norm(bc) if np.linalg.norm(bc) != 0 else bc

        cosine_angle = np.dot(ba_norm, bc_norm)
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle_rad)

    def get_finger_angles(self, points_list, calc_func, landmarks):
        """
        :param points_list: 关节点索引列表，如 [[0,1,2], [1,2,3]]
        :param calc_func: calculate_angle_2d / calculate_angle_3d
        :param landmarks: 实际点坐标列表，如 [(x,y), (x,y), ...]
        :return: list of angles
        """
        return [
            calc_func(landmarks[i], landmarks[j], landmarks[k])
            for i, j, k in points_list
        ]

    def get_all_finger_angles(self, landmarks, use_2d=True):
        """
        获取所有手指的角度（每根手指 3 个角度）
        :param landmarks: 手部关键点（2D 或 3D）
        :param use_2d: 是否使用 2D 角度计算
        :return: dict 包含每根手指的角度
        """
        calc_func = self.calculate_angle_2d if use_2d else self.calculate_angle_3d
        raw_angles = {
            finger: self.get_finger_angles(joints, calc_func, landmarks)
            for finger, joints in FINGER_JOINTS.items()
        }
    
        # 转换为“伸直=0，弯曲越大数值越大”
        bend_angles = {
            finger: [180 - round(angle, 1) for angle in angles]
            for finger, angles in raw_angles.items()
        }

        return bend_angles
   
    def draw_landmarks_on_image(self, frame, hand_landmarks, angles=None):
        """在图像上绘制手部关键点、连接线 和 编号"""
        frame_with_landmarks = frame.copy()

        # --- 使用 MediaPipe 默认绘图方式（红色）---
        self.mp_drawing.draw_landmarks(
            frame_with_landmarks,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

        # 获取图像尺寸
        h, w, _ = frame_with_landmarks.shape

        if angles is not None:
            for finger_name, joint_angles in angles.items():
                joint_indices = FINGER_JOINTS[finger_name]
                for i, indices in enumerate(joint_indices):
                    if i >= len(joint_angles):
                        continue
                    try:
                        a_idx, b_idx, c_idx = indices
                        b = hand_landmarks.landmark[b_idx]
                        x = int(b.x * w)
                        y = int(b.y * h)

                        angle = round(joint_angles[i], 1)

                        # 调整字体大小为 0.4（原为 0.5）
                        font_scale = 0.4
                        thickness = 1

                         # 白色描边 + 黄色字体（更清晰）
                        cv2.putText(frame_with_landmarks, f"{angle}",
                                    (x + 8, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale,
                                    (255, 255, 255),  # 白色描边
                                    thickness + 1,
                                    cv2.LINE_AA)

                        cv2.putText(frame_with_landmarks, f"{angle}",
                                    (x + 8, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale,
                                    (0, 255, 255),  # 黄色字体
                                    thickness,
                                    cv2.LINE_AA)
                    except Exception as e:
                        print(f"[ERROR] Drawing angle failed for {finger_name} joint {i}: {e}")


        return frame_with_landmarks


    def apply_exponential_smoothing(self, current_landmarks):
        """指数平滑滤波函数"""
        if self.previous_landmarks is None:
            self.previous_landmarks = current_landmarks
            return current_landmarks
        
        # 计算平滑后的关键点
        smoothed_landmarks = (
            self.smoothing_factor * current_landmarks +
            (1 - self.smoothing_factor) * self.previous_landmarks
        )
        self.previous_landmarks = smoothed_landmarks
        return smoothed_landmarks


    def initialize_opencv_kalman_filters(self):
        """使用OpenCV的卡尔曼滤波器实现"""
        for i in range(21):
            # X坐标滤波器
            kf_x = cv2.KalmanFilter(2, 1)  # 2状态变量，1测量值
            kf_x.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
            kf_x.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
            kf_x.processNoiseCov = np.eye(2, dtype=np.float32) * 0.1
            kf_x.measurementNoiseCov = np.array([[1]], dtype=np.float32)
            kf_x.statePost = np.array([[0], [0]], dtype=np.float32)  # 初始状态
            
            # Y坐标滤波器
            kf_y = cv2.KalmanFilter(2, 1)
            kf_y.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
            kf_y.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
            kf_y.processNoiseCov = np.eye(2, dtype=np.float32) * 0.1
            kf_y.measurementNoiseCov = np.array([[1]], dtype=np.float32)
            kf_y.statePost = np.array([[0], [0]], dtype=np.float32)
            
            self.kalman_filters[f'x_{i}'] = kf_x
            self.kalman_filters[f'y_{i}'] = kf_y


    def apply_kalman_filter(self, landmarks):
        """应用OpenCV卡尔曼滤波"""
        smoothed_landmarks = np.zeros_like(landmarks)
        
        for i in range(len(landmarks)):
            x, y = landmarks[i]
            
            # 预测
            self.kalman_filters[f'x_{i}'].predict()
            self.kalman_filters[f'y_{i}'].predict()
            
            # 更新
            smoothed_x = self.kalman_filters[f'x_{i}'].correct(np.array([[x]], dtype=np.float32))
            smoothed_y = self.kalman_filters[f'y_{i}'].correct(np.array([[y]], dtype=np.float32))
            
            smoothed_landmarks[i] = [smoothed_x[0][0], smoothed_y[0][0]]
            
        return smoothed_landmarks
    

    def process_frame(self, frame, results):
        angles = []
        annotated_image = frame

        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            for idx, (hand_landmarks, hand_world_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks,
                        results.multi_hand_world_landmarks,
                        results.multi_handedness)):

                # 提取当前帧的关键点坐标（21个点，每个点有x, y）
                current_landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                
                # 应用指数平滑滤波
                smoothed_landmarks = self.apply_exponential_smoothing(current_landmarks)
                
                # 更新手部关键点数据（使用平滑后的坐标）
                for i, lm in enumerate(hand_landmarks.landmark):
                    lm.x, lm.y = smoothed_landmarks[i]
                    

                use_2d = True  # 可切换使用 3D

                if use_2d:
                    landmarks_2d = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    angles = self.get_all_finger_angles(landmarks_2d, use_2d=True)
                else:
                    landmarks_3d = [(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark]
                    angles = self.get_all_finger_angles(landmarks_3d, use_2d=False)


                # 绘制图像
                annotated_image = self.draw_landmarks_on_image(frame, hand_landmarks, angles)

                # 获取左右手信息
                hand_label = handedness.classification[0].label

                # 构造元数据
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "hand_index": idx,
                    "hand_label": hand_label,
                    "angles": angles
                }


                # 编码图像
                _, jpeg = cv2.imencode('.jpg', annotated_image)

                # 发送 multipart 数据
                self.socket.send_multipart([
                    json.dumps(metadata).encode('utf-8'),
                    jpeg.tobytes()
                ])

        else:
            # 没有检测到手时，发送空的数据和原图
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "hand_index": None,
                "hand_label": None,
                "angles": angles
            }

            # 使用原始帧图像（未标注）
            _, jpeg = cv2.imencode('.jpg', frame)

            # 发送 multipart 数据
            self.socket.send_multipart([
                json.dumps(metadata, ensure_ascii=False).encode('utf-8'),
                jpeg.tobytes()
            ])

        # 调用显示函数（传入带关键点的图像和角度数据）
        if self.show_window:
            show_simple(annotated_image, angles)


    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_index}")
            return

        print(f"[INFO] Using camera index: {camera_index}")
        print("[INFO] hand gesure detector is running")
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.flip(frame_rgb, 1)
                results = self.hands.process(frame_rgb)

                # 处理并发送数据
                self.process_frame(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), results)

                if cv2.waitKey(1) == 27 or not self.running:
                    break
        finally:
            cap.release()
            self.socket.close()
            self.context.term()
            print("[INFO] Resources released.")


if __name__ == '__main__':
    detector = HandGestureDetector()
    detector.set_show_window(True)
    detector.run(camera_index=0)