import cv2
import mediapipe as mp
import numpy as np
import zmq
import json
from datetime import datetime
import os
import sys
import signal

def resource_path(relative_path):
    """ 获取资源绝对路径，适用于 PyInstaller 打包后的环境 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class HandGestureDetector:
    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        # 设置 Mediapipe 自定义模型路径（打包后也能找到）
        mp_path = resource_path('mediapipe')

        os.environ['MEDIAPIPE_HOME'] = mp_path

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # 开启 model_complexity=1 以启用 3D 坐标
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        # ZeroMQ 设置
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")  # 监听端口 5555
        print("[INFO] ZMQ Publisher started on tcp://*:5555")


    def exit_gracefully(self, *args):
        print("[INFO] Exiting gracefully...")
        self.running = False

    def calculate_angle_3d(self, a, b, c):
        """
        输入三个 3D 点 a, b, c，计算 ∠abc 的角度（单位为度）
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        # 单位向量
        ba_norm = ba / np.linalg.norm(ba) if np.linalg.norm(ba) != 0 else ba
        bc_norm = bc / np.linalg.norm(bc) if np.linalg.norm(bc) != 0 else bc

        # 夹角（弧度）
        cosine_angle = np.dot(ba_norm, bc_norm)
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle_rad)

    def get_finger_angles_3d(self, world_landmarks):
        """
        输入一个手的 3D 关键点列表，返回每根手指各关节的弯曲角度
        """
        landmarks = [(lm.x, lm.y, lm.z) for lm in world_landmarks]

        angles = {}

        # thumb
        angles['thumb'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[1], landmarks[2]),
            self.calculate_angle_3d(landmarks[1], landmarks[2], landmarks[3]),
            self.calculate_angle_3d(landmarks[2], landmarks[3], landmarks[4])
        ]

        # index
        angles['index'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[5], landmarks[6]),
            self.calculate_angle_3d(landmarks[5], landmarks[6], landmarks[7]),
            self.calculate_angle_3d(landmarks[6], landmarks[7], landmarks[8])
        ]

        # middle
        angles['middle'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[9], landmarks[10]),
            self.calculate_angle_3d(landmarks[9], landmarks[10], landmarks[11]),
            self.calculate_angle_3d(landmarks[10], landmarks[11], landmarks[12])
        ]

        # ring
        angles['ring'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[13], landmarks[14]),
            self.calculate_angle_3d(landmarks[13], landmarks[14], landmarks[15]),
            self.calculate_angle_3d(landmarks[14], landmarks[15], landmarks[16])
        ]

        # pinky
        angles['pinky'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[17], landmarks[18]),
            self.calculate_angle_3d(landmarks[17], landmarks[18], landmarks[19]),
            self.calculate_angle_3d(landmarks[18], landmarks[19], landmarks[20])
        ]

        converted_angles = {
            finger: [180 - a for a in joint_angles]
            for finger, joint_angles in angles.items()
        }

        return converted_angles

    def draw_landmarks_on_image(self, frame, hand_landmarks):
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

        # --- 绘制编号（0~20）+ 白色描边 ---
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            # 先画白色描边（较粗）
            cv2.putText(
                frame_with_landmarks, str(idx),
                (x + 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),  # 白色描边
                2,  # 线条粗细
                cv2.LINE_AA
            )

            # 再画内部红色数字（稍细）
            cv2.putText(
                frame_with_landmarks, str(idx),
                (x + 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),  # 红色字体
                1,
                cv2.LINE_AA
            )

        return frame_with_landmarks

    def process_frame(self, frame, results):
        data_list = []

        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            for idx, (hand_landmarks, hand_world_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks,
                        results.multi_hand_world_landmarks,
                        results.multi_handedness)):

                # 获取世界坐标
                landmarks_3d = [{
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                } for lm in hand_world_landmarks.landmark]

                # 获取角度
                angles = self.get_finger_angles_3d(hand_world_landmarks.landmark)

                # print(angles)  # 应该是 {'thumb': [...], ...}

                # 获取左右手信息
                hand_label = handedness.classification[0].label

                # 构造元数据
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "hand_index": idx,
                    "hand_label": hand_label,
                    "landmarks_3d": landmarks_3d,
                    "angles": angles
                }


                # 绘制图像
                annotated_image = self.draw_landmarks_on_image(frame, hand_landmarks)

                # 编码图像
                _, jpeg = cv2.imencode('.jpg', annotated_image)

                # 发送 multipart 数据
                self.socket.send_multipart([
                    json.dumps(metadata).encode('utf-8'),
                    jpeg.tobytes()
                ])

                data_list.append(metadata)
        else:
            # 没有检测到手时，发送空的数据和原图
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "hand_index": None,
                "hand_label": None,
                "landmarks_3d": [],
                "angles": {}
            }

            # 使用原始帧图像（未标注）
            _, jpeg = cv2.imencode('.jpg', frame)

            # 发送 multipart 数据
            self.socket.send_multipart([
                json.dumps(metadata, ensure_ascii=False).encode('utf-8'),
                jpeg.tobytes()
            ])

            data_list.append(metadata)

        return data_list

    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_index}")
            return

        print(f"[INFO] Using camera index: {camera_index}")
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
    detector.run(camera_index=0)