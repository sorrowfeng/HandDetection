import cv2
import mediapipe as mp
import math
import yaml

class HandGestureDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.angle_records = []

    def vector_2d_angle(self, v1, v2):
        '''
            求解二维向量的角度
        '''
        v1_x = v1[0]
        v1_y = v1[1]
        v2_x = v2[0]
        v2_y = v2[1]
        try:
            angle_ = math.degrees(math.acos((v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
        except:
            angle_ = 65535.
        if angle_ > 180.:
            angle_ = 65535.
        return angle_

    def hand_angle(self, hand_):
        '''
            获取对应手相关向量的二维角度,根据角度确定手势
        '''
        angle_list = []
        #---------------------------- thumb 大拇指角度
        vector1_thumb = (int(hand_[0][0]) - int(hand_[2][0]), int(hand_[0][1]) - int(hand_[2][1]))
        vector2_thumb = (int(hand_[3][0]) - int(hand_[4][0]), int(hand_[3][1]) - int(hand_[4][1]))
        angle_ = self.vector_2d_angle(vector1_thumb, vector2_thumb)
        angle_list.append(angle_)
        #---------------------------- thumb 大拇指弯曲角度
        angle_list.append(0)
        #---------------------------- index 食指角度
        vector1_index = (int(hand_[0][0]) - int(hand_[6][0]), int(hand_[0][1]) - int(hand_[6][1]))
        vector2_index = (int(hand_[7][0]) - int(hand_[8][0]), int(hand_[7][1]) - int(hand_[8][1]))
        angle_ = self.vector_2d_angle(vector1_index, vector2_index)
        angle_list.append(angle_)
        #---------------------------- middle 中指角度
        vector1_middle = (int(hand_[0][0]) - int(hand_[10][0]), int(hand_[0][1]) - int(hand_[10][1]))
        vector2_middle = (int(hand_[11][0]) - int(hand_[12][0]), int(hand_[11][1]) - int(hand_[12][1]))
        angle_ = self.vector_2d_angle(vector1_middle, vector2_middle)
        angle_list.append(angle_)
        #---------------------------- ring 无名指角度
        vector1_ring = (int(hand_[0][0]) - int(hand_[14][0]), int(hand_[0][1]) - int(hand_[14][1]))
        vector2_ring = (int(hand_[15][0]) - int(hand_[16][0]), int(hand_[15][1]) - int(hand_[16][1]))
        angle_ = self.vector_2d_angle(vector1_ring, vector2_ring)
        angle_list.append(angle_)
        #---------------------------- pink 小拇指角度
        vector1_pink = (int(hand_[0][0]) - int(hand_[18][0]), int(hand_[0][1]) - int(hand_[18][1]))
        vector2_pink = (int(hand_[19][0]) - int(hand_[20][0]), int(hand_[19][1]) - int(hand_[20][1]))
        angle_ = self.vector_2d_angle(vector1_pink, vector2_pink)
        angle_list.append(angle_)
        return angle_list

    def draw_hand_info(self, frame, results):
        if results.multi_handedness:
            for hand_label in results.multi_handedness:
                hand_jugg = str(hand_label).split('"')[1]
                cv2.putText(frame, hand_jugg, (50, 200), 0, 1.3, (0, 0, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame.shape[1]
                    y = hand_landmarks.landmark[i].y * frame.shape[0]
                    hand_local.append((x, y))

                if hand_local:
                    angle_list = self.hand_angle(hand_local)
                    x_coord = frame.shape[1] - 300
                    y_coord = 50
                    for i, angle in enumerate(angle_list):
                        angle_text = f"Joint {i + 1} Angle: {angle:.2f}"
                        cv2.putText(frame, angle_text, (x_coord, y_coord), 0, 0.7, (0, 0, 255), 1)
                        y_coord += 30

                    return angle_list
        return None

    def export_to_yaml(self):
        if self.angle_records is None or len(self.angle_records) == 0:
            print("没有要导出的数据")
            return
                    
        # 假设在类的 __init__ 方法中已经定义了 self.max_angles 成员变量
        # 这里使用 self.max_angles 来处理角度数据
        max_angles = getattr(self, 'max_angles', [110, 60, 84, 90, 90, 88])
        action_sequence = {"action_sequence": []}
        for angles in self.angle_records:
            processed_angles = [min(angle, max_angles[i]) if i < len(max_angles) else angle for i, angle in enumerate(angles)]
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            results = self.hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            angle_list = self.draw_hand_info(frame, results)

            cv2.imshow('MediaPipe Hands', frame)
            key = cv2.waitKey(1)
            if key == 13 and angle_list is not None:  # 按回车记录角度
                self.angle_records.append(angle_list)
            elif key & 0xFF == 27:  # 按 ESC 退出
                break

        cap.release()
        cv2.destroyAllWindows()

        # 直接调用 export_to_yaml 方法导出记录的角度数据
        if self.angle_records:
            self.export_to_yaml()

if __name__ == '__main__':
    detector = HandGestureDetector()
    detector.detect()