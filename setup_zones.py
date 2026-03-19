"""
Zone Setup Tool
===============
Run this script once to interactively define the arm polygons for your footage.
Click polygon vertices on the first frame of the video, press ENTER to record
each arm, and copy the printed numpy arrays into ARM_POLYGONS in main.py.

Usage:
    python setup_zones.py
    python setup_zones.py --video roundabout.mp4
"""

import cv2
import numpy as np

from main import DEFAULT_SOURCE_VIDEO, ARM_ORDER


def setup_zones_interactively(video_path: str) -> None:
    """
    Opens the first frame of the video and lets you click to define polygon
    vertices for each arm. Press ENTER to finish a polygon, ESC to quit.
    Prints the resulting numpy arrays to copy into ARM_POLYGONS.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read video.")
        return

    points: list[tuple[int, int]] = []
    clone = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow("Zone Setup", frame)

    cv2.namedWindow("Zone Setup")
    cv2.setMouseCallback("Zone Setup", mouse_callback)
    cv2.imshow("Zone Setup", frame)

    print("Click to define polygon vertices. Press ENTER to record, ESC to quit.")
    arm_idx = 0
    arm_names = ARM_ORDER

    while arm_idx < len(arm_names):
        print(f"\nDefine zone for: {arm_names[arm_idx]}")
        points.clear()
        frame = clone.copy()
        cv2.imshow("Zone Setup", frame)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(points) >= 3:  # ENTER
                arr = np.array(points)
                print(f'    "{arm_names[arm_idx]}": np.array({arr.tolist()}),')
                arm_idx += 1
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\nCopy the above into ARM_POLYGONS in main.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive zone setup tool")
    parser.add_argument("--video", default=DEFAULT_SOURCE_VIDEO)
    args = parser.parse_args()
    setup_zones_interactively(args.video)
