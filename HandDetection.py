import cv2
import mediapipe as mp
import numpy as np
import yaml


class HandGestureDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # 开启 model_complexity=1 以启用 3D 坐标
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,  # 关键点：启用 3D 坐标
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.angle_records = []

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
        返回格式：
            {
                'thumb': [angle1, angle2],
                'index': [angle1, angle2],
                ...
            }
        """
        landmarks = [
            (lm.x, lm.y, lm.z) for lm in world_landmarks
        ]

        angles = {}

        # thumb
        angles['thumb'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[1], landmarks[2]),  # WRIST -> CMC -> MCP
            self.calculate_angle_3d(landmarks[1], landmarks[2], landmarks[3]),  # CMC -> MCP -> IP
            self.calculate_angle_3d(landmarks[2], landmarks[3], landmarks[4])   # MCP -> IP -> TIP
        ]

        # index
        angles['index'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[5], landmarks[6]),  # WRIST -> MCP -> PIP
            self.calculate_angle_3d(landmarks[5], landmarks[6], landmarks[7]),  # MCP -> PIP -> DIP
            self.calculate_angle_3d(landmarks[6], landmarks[7], landmarks[8])   # PIP -> DIP -> TIP
        ]

        # middle
        angles['middle'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[9], landmarks[10]),  # WRIST -> MCP -> PIP
            self.calculate_angle_3d(landmarks[9], landmarks[10], landmarks[11]), # MCP -> PIP -> DIP
            self.calculate_angle_3d(landmarks[10], landmarks[11], landmarks[12]) # PIP -> DIP -> TIP
        ]

        # ring
        angles['ring'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[13], landmarks[14]),  # WRIST -> MCP -> PIP
            self.calculate_angle_3d(landmarks[13], landmarks[14], landmarks[15]), # MCP -> PIP -> DIP
            self.calculate_angle_3d(landmarks[14], landmarks[15], landmarks[16])  # PIP -> DIP -> TIP
        ]

        # pinky
        angles['pinky'] = [
            self.calculate_angle_3d(landmarks[0], landmarks[17], landmarks[18]),  # WRIST -> MCP -> PIP
            self.calculate_angle_3d(landmarks[17], landmarks[18], landmarks[19]), # MCP -> PIP -> DIP
            self.calculate_angle_3d(landmarks[18], landmarks[19], landmarks[20])  # PIP -> DIP -> TIP
        ]

        converted_angles = {
            finger: [180 - a for a in angles]
            for finger, angles in angles.items()
        }

        return converted_angles

    def draw_hand_info(self, frame, results):
        if results.multi_handedness:
            for hand_label in results.multi_handedness:
                hand_jugg = str(hand_label).split('"')[1]
                cv2.putText(frame, f'Hand: {hand_jugg}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)

        all_angles = []

        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            # 同时遍历图像坐标和世界坐标
            for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks,
                                                            results.multi_hand_world_landmarks):
                # --- 绘制图像坐标点 ---
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # 获取图像坐标（用于绘图）
                image_coords = []
                for i in range(21):
                    x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[i].y * frame.shape[0])
                    image_coords.append((x, y))
                    # 画出编号
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (x + 5, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                # --- 使用世界坐标计算角度 ---
                angles_dict = self.get_finger_angles_3d(hand_world_landmarks.landmark)

                # --- 显示在图像上 ---
                start_x, start_y = frame.shape[1] - 300, 50
                line_spacing = 30

                # 控制台打印
                for finger, angles in angles_dict.items():
                    angle_str = ", ".join([f"{a:.1f}" for a in angles])
                    print(f"{finger.capitalize()}: {angle_str}")

                # 图像上显示
                for finger, angles in angles_dict.items():
                    angle_str = ", ".join([f"{a:.1f}" for a in angles])
                    text = f"{finger.capitalize()}: {angle_str}"
                    cv2.putText(frame, text, (start_x, start_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    start_y += line_spacing

                # 返回所有角度用于记录
                all_angles_list = []
                for angles in angles_dict.values():
                    all_angles_list.extend(angles)

                return all_angles_list

        return None

    def export_to_yaml(self):
        if not self.angle_records:
            print("没有要导出的数据")
            return

        action_sequence = {"action_sequence": []}
        for angles in self.angle_records:
            processed_angles = [min(angle, 180) for angle in angles]
            joint_ids = list(range(1, len(processed_angles) + 1))
            action_sequence["action_sequence"].append({
                "joint_ids": joint_ids,
                "angles": processed_angles
            })

        with open('action_sequence.yaml', 'w') as f:
            yaml.dump(action_sequence, f, default_flow_style=False, sort_keys=False)
        print("数据已导出到 action_sequence.yaml 文件")

    def detect(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            results = self.hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            angle_list = self.draw_hand_info(frame, results)

            cv2.imshow('MediaPipe Hands (3D Angle)', frame)
            key = cv2.waitKey(1)

            if key == 13 and angle_list is not None:  # 按回车记录角度
                self.angle_records.append(angle_list)
            elif key == 27:  # 按 ESC 退出
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.angle_records:
            self.export_to_yaml()


if __name__ == '__main__':
    detector = HandGestureDetector()
    detector.detect()