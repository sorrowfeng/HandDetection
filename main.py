# main.py
from hand_gesture_detector import HandGestureDetector
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture Detector")
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index (default: 0)')
    args = parser.parse_args()

    detector = HandGestureDetector()
    detector.run(camera_index=args.camera)