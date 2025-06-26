# main.py
from hand_gesture_detector import HandGestureDetector
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture Detector")
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--show-window', action='store_true', help='Show the video window (default: False)')

    args = parser.parse_args()

    detector = HandGestureDetector()
    detector.set_show_window(args.show_window)  # 设置是否显示窗口
    detector.run(camera_index=args.camera)