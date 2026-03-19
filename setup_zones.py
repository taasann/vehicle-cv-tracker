"""
Zone Setup Tool
===============
Run this script once to interactively define the arm polygons for your footage.
Click polygon vertices on the first frame of the video, press ENTER to record
each arm. The resulting zones are saved to project.json (or a custom path via
--project) and loaded automatically by tracker.py.

Usage:
    python setup_zones.py
    python setup_zones.py --video roundabout.mp4
    python setup_zones.py --project custom.json
"""

import argparse
import json

import cv2

from tracker import DEFAULT_SOURCE_VIDEO, PROJECT_FILE

DEFAULT_ARM_NAMES = ["North", "East", "South", "West"]

# 10 perceptually distinct colors for zone annotation
ZONE_COLOR_PALETTE = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#00CED1",  # teal
    "#F781BF",  # pink
    "#A65628",  # brown
    "#FFD700",  # gold
    "#808080",  # gray
]


def setup_zones_interactively(video_path: str, project_path: str) -> None:
    """
    Opens the first frame of the video and lets you click to define polygon
    vertices for each arm. Press ENTER to finish a polygon, ESC to quit.
    Saves the resulting zones to project_path as JSON.
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
    arm_names = DEFAULT_ARM_NAMES
    zones: dict[str, dict] = {}

    while arm_idx < len(arm_names):
        color = ZONE_COLOR_PALETTE[arm_idx % len(ZONE_COLOR_PALETTE)]
        print(f"\nDefine zone for: {arm_names[arm_idx]} (color: {color})")
        points.clear()
        frame = clone.copy()
        cv2.imshow("Zone Setup", frame)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(points) >= 3:  # ENTER
                zones[arm_names[arm_idx]] = {
                    "polygon": [list(p) for p in points],
                    "color": color,
                }
                arm_idx += 1
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    with open(project_path, "w") as f:
        json.dump({"zones": zones}, f, indent=2)
    print(f"\nSaved zones to {project_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive zone setup tool")
    parser.add_argument("--video", default=DEFAULT_SOURCE_VIDEO)
    parser.add_argument("--project", default=PROJECT_FILE, help="Output project JSON path")
    args = parser.parse_args()
    setup_zones_interactively(args.video, args.project)
