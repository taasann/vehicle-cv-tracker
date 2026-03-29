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
import numpy as np

from tracker import DEFAULT_SOURCE_VIDEO, PROJECT_FILE

DEFAULT_ARM_NAMES = ["Linn", "Jarveotsa", "Paldiski"]

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


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_status_bar(
    frame: np.ndarray,
    arm_name: str,
    arm_color_bgr: tuple[int, int, int],
    n_points: int,
    arm_idx: int,
    total_arms: int,
    polygon_role: str = "Zone",
) -> None:
    """Render a two-line status bar at the bottom of frame (in-place)."""
    h, w = frame.shape[:2]
    bar_height = 60
    pad = 8

    # Semi-transparent dark backdrop
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Line 1: color swatch + arm name + progress
    swatch_x, swatch_y = pad, h - bar_height + pad
    cv2.rectangle(frame, (swatch_x, swatch_y), (swatch_x + 16, swatch_y + 16), arm_color_bgr, -1)
    label = (
        f"Zone {arm_idx + 1}/{total_arms}: {arm_name} — {polygon_role} polygon"
        f"   ({n_points} point{'s' if n_points != 1 else ''} placed)"
    )
    cv2.putText(frame, label, (swatch_x + 22, swatch_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, arm_color_bgr, 1, cv2.LINE_AA)

    # Line 2: instructions
    hint = "Click: add vertex    ENTER: confirm polygon (need >=3)    ESC: quit"
    cv2.putText(frame, hint, (pad, h - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_current_polygon(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    color_bgr: tuple[int, int, int],
) -> None:
    """Draw in-progress polygon points and edges onto frame (in-place)."""
    for i, pt in enumerate(points):
        cv2.circle(frame, pt, 5, color_bgr, -1)
        if i > 0:
            cv2.line(frame, points[i - 1], pt, color_bgr, 2)
    # Close the polygon preview once we have enough points
    if len(points) >= 3:
        cv2.line(frame, points[-1], points[0], color_bgr, 1)


def draw_completed_zone(
    background: np.ndarray,
    pts: list[list[int]],
    color_bgr: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a finished zone outline onto the background frame (in-place)."""
    poly = np.array(pts, dtype=np.int32)
    cv2.polylines(background, [poly], isClosed=True, color=color_bgr, thickness=thickness)


def get_display_scale(frame_w: int, frame_h: int) -> float:
    """Return a scale factor (<= 1.0) so the frame fits within the screen."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception:
        screen_w, screen_h = 1920, 1080
    # Leave some headroom for window chrome / taskbar
    margin = 0.92
    return min(screen_w * margin / frame_w, screen_h * margin / frame_h, 1.0)



def setup_zones_interactively(video_path: str, project_path: str) -> None:
    """
    Opens the first frame of the video and lets you click to define polygon
    vertices for each arm. Press ENTER to finish a polygon, ESC to quit.
    Saves the resulting zones to project_path as JSON.

    The window is scaled to fit within the display, but stored coordinates are
    always in the original video resolution.
    """
    cap = cv2.VideoCapture(video_path)
    ret, base_frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read video.")
        return

    frame_h, frame_w = base_frame.shape[:2]
    scale = get_display_scale(frame_w, frame_h)
    display_w = int(frame_w * scale)
    display_h = int(frame_h * scale)

    # background accumulates completed zone outlines (original resolution)
    background = base_frame.copy()

    # points stored in original resolution
    points: list[tuple[int, int]] = []
    arm_idx = 0
    arm_names = DEFAULT_ARM_NAMES
    zones: dict[str, dict] = {}

    # current drawing state (mutated by mouse callback)
    state = {"redraw": False}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coordinates back to original resolution
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            points.append((orig_x, orig_y))
            state["redraw"] = True

    cv2.namedWindow("Zone Setup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zone Setup", display_w, display_h)
    cv2.setMouseCallback("Zone Setup", mouse_callback)

    while arm_idx < len(arm_names):
        color_hex = ZONE_COLOR_PALETTE[arm_idx % len(ZONE_COLOR_PALETTE)]
        color_bgr = hex_to_bgr(color_hex)
        arm_name = arm_names[arm_idx]
        arm_polygons: dict[str, list] = {}

        for polygon_role in ("Entry", "Exit"):
            points.clear()
            state["redraw"] = True

            while True:
                if state["redraw"]:
                    frame = background.copy()
                    draw_current_polygon(frame, points, color_bgr)
                    draw_status_bar(frame, arm_name, color_bgr,
                                    len(points), arm_idx, len(arm_names),
                                    polygon_role=polygon_role)
                    display = cv2.resize(frame, (display_w, display_h))
                    cv2.imshow("Zone Setup", display)
                    state["redraw"] = False

                key = cv2.waitKey(20) & 0xFF
                if key == 13 and len(points) >= 3:  # ENTER
                    arm_polygons[polygon_role] = [list(p) for p in points]
                    thickness = 2 if polygon_role == "Entry" else 1
                    draw_completed_zone(background, arm_polygons[polygon_role], color_bgr, thickness)
                    break
                elif key == 27:  # ESC
                    cv2.destroyAllWindows()
                    return

        zones[arm_name] = {
            "entry_polygon": arm_polygons["Entry"],
            "exit_polygon": arm_polygons["Exit"],
            "color": color_hex,
        }
        arm_idx += 1

    with open(project_path, "w") as f:
        json.dump({"zones": zones}, f, indent=2)

    # Show save confirmation in the window
    frame = background.copy()
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"Saved zones to {project_path}",
                (8, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press any key to close.",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    display = cv2.resize(frame, (display_w, display_h))
    cv2.imshow("Zone Setup", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive zone setup tool")
    parser.add_argument("--video", default=DEFAULT_SOURCE_VIDEO)
    parser.add_argument("--project", default=PROJECT_FILE, help="Output project JSON path")
    args = parser.parse_args()
    setup_zones_interactively(args.video, args.project)
