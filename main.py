"""
Roundabout Vehicle Tracker
==========================
Tracks vehicles in drone footage of a roundabout and records results based on
which arm each vehicle enters and exits from.

Requirements:
    pip install ultralytics supervision opencv-python numpy

Usage:
    python roundabout_tracker.py --source footage.mp4 --output output.mp4
"""

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_VIDEO = "footage.mp4"
OUTPUT_VIDEO = "output.mp4"
MODEL_PATH   = "yolo26x.pt"

# COCO class IDs for vehicles (car, motorcycle, bus, truck)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.1

# How many frames a vehicle must be absent before its journey is "closed"
TRACK_TIMEOUT_FRAMES = 120


# ---------------------------------------------------------------------------
# Roundabout Geometry
# ---------------------------------------------------------------------------
# Define one polygon per road arm of the roundabout.
# Each polygon should cover the entry AND exit lanes of that arm.
#
# !! You must update these coordinates to match your footage. !!
# Tip: Run `python setup_zones.py` (see bottom of file) to pick coordinates
# interactively by clicking on a reference frame.
#
# Arm ordering matters for maneuver classification — keep it consistent.
# Example layout (4-arm roundabout, arms numbered clockwise from North):
#
#          ARM 0 (North)
#              |
#  ARM 3 ---- O ---- ARM 1 (East)
#  (West)     |
#          ARM 2 (South)

ARM_POLYGONS = {
    "NE": np.array([[973, 260], [1200, 440], [1312, 221], [1162, 129], [1096, 218], [974, 260]]),
    "SE": np.array([[1233, 554], [1099, 754], [1181, 764], [1318, 879], [1417, 776], [1336, 701], [1268, 624], [1235, 556]]),
    "S": np.array([[679, 758], [866, 761], [927, 762], [1027, 800], [970, 855], [949, 918], [942, 1011], [797, 1012], [797, 881], [780, 837], [743, 794], [679, 760]]),
    "W": np.array([[727, 506], [766, 657], [686, 763], [623, 758], [468, 769], [456, 642], [574, 628], [634, 603], [725, 513]]),
    "NW": np.array([[559, 257], [648, 147], [794, 244], [913, 274], [809, 355], [721, 438], [685, 354], [560, 260]]),
}

ZONE_COLORS = ["#FF5733", "#33FF57", "#3357FF", "#FF33F5", "#FF0000"]

# Clockwise ordering of arms — used to determine turn direction
ARM_ORDER = ["North", "East", "South", "West", "Extra"]



# ---------------------------------------------------------------------------
# Vehicle Journey Tracking
# ---------------------------------------------------------------------------

@dataclass
class VehicleJourney:
    """Stores the entry/exit state for a single tracked vehicle."""
    track_id: int
    entry_arm: str | None = None
    exit_arm:  str | None = None
    last_seen_frame: int  = 0
    completed: bool       = False
    positions: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main Tracker
# ---------------------------------------------------------------------------

class RoundaboutTracker:
    def __init__(self, source: str, output: str):
        self.source_path = source
        self.output_path = output

        # Detection model
        self.model = YOLO(MODEL_PATH)

        # ByteTrack tracker
        self.tracker = sv.ByteTrack()

        # Build PolygonZone objects for each arm
        self.zones: dict[str, sv.PolygonZone] = {}
        for arm_name, polygon in ARM_POLYGONS.items():
            self.zones[arm_name] = sv.PolygonZone(polygon=polygon)

        # Per-vehicle journey state
        self.journeys: dict[int, VehicleJourney] = {}

        # Annotators for visualisation
        self.box_annotator   = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.zone_annotators = {
            name: sv.PolygonZoneAnnotator(
                zone=zone,
                color=sv.Color.from_hex(color),
                thickness=2,
            )
            for (name, zone), color in zip(
                self.zones.items(),
                ZONE_COLORS,
            )
        }

        # BGR color per arm for path drawing
        self.arm_colors: dict[str, tuple[int, int, int]] = {
            name: tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
            for name, hex_color in zip(ARM_POLYGONS.keys(), ZONE_COLORS)
        }

    # ------------------------------------------------------------------
    # Zone Logic
    # ------------------------------------------------------------------

    def _arm_containing(self, detections: sv.Detections) -> dict[int, str]:
        """Return a mapping of {track_id: arm_name} for detected vehicles."""
        result = {}
        for arm_name, zone in self.zones.items():
            mask = zone.trigger(detections=detections)
            for i, in_zone in enumerate(mask):
                if in_zone and detections.tracker_id is not None:
                    tid = int(detections.tracker_id[i])
                    result[tid] = arm_name
        return result

    # ------------------------------------------------------------------
    # Journey State Machine
    # ------------------------------------------------------------------

    def _update_journeys(
        self,
        arm_map: dict[int, str],
        active_ids: set[int],
        frame_idx: int,
    ) -> None:
        """
        Update each vehicle's journey state based on which zone it's in.

        State transitions:
          - Vehicle enters a zone and has no entry arm → record entry arm
          - Vehicle enters a zone AND already has an entry arm AND the zone is
            different → record exit arm and finalise maneuver
        """
        # Initialise journeys for new track IDs
        for tid in active_ids:
            if tid not in self.journeys:
                self.journeys[tid] = VehicleJourney(track_id=tid)
            self.journeys[tid].last_seen_frame = frame_idx

        for tid, arm in arm_map.items():
            journey = self.journeys.get(tid)
            if journey is None or journey.completed:
                continue

            if journey.entry_arm is None:
                # First arm contact → record as entry
                journey.entry_arm = arm

            elif arm != journey.entry_arm and journey.exit_arm is None:
                # Different arm → record as exit and classify
                journey.exit_arm = arm
                journey.completed = True
                print(
                    f"[Frame {frame_idx}] Vehicle {tid}: "
                    f"{journey.entry_arm} → {journey.exit_arm} "
                )

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _build_labels(self, detections: sv.Detections) -> list[str]:
        labels = []
        for tid in (detections.tracker_id if detections.tracker_id is not None else []):
            tid = int(tid)
            journey = self.journeys.get(tid)
            if journey and journey.completed:
                labels.append(f"#{tid} from {journey.entry_arm} to {journey.exit_arm}")
            elif journey and journey.entry_arm:
                labels.append(f"#{tid} from {journey.entry_arm}")
            else:
                labels.append(f"#{tid}")
        return labels

    def _draw_paths(self, frame: np.ndarray, frame_idx) -> np.ndarray:
        for journey in self.journeys.values():
            pts = journey.positions
            if len(pts) < 2 or frame_idx - journey.last_seen_frame > TRACK_TIMEOUT_FRAMES:
                continue
            for j in range(1, len(pts)):
                if journey.entry_arm is None:
                    color = (0, 255, 255)
                else:
                    color = self.arm_colors.get(journey.entry_arm, (0, 255, 255))

                cv2.line(frame, pts[j-1], pts[j], color, 2)
        return frame

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cap    = cv2.VideoCapture(self.source_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_idx = 0
        start_time = time.time()
        print(f"Processing {self.source_path} ...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Detection ---
            results = self.model(
                frame,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                classes=VEHICLE_CLASS_IDS,
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            # --- Tracking ---
            detections = self.tracker.update_with_detections(detections)

            # --- Zone Logic ---
            arm_map    = self._arm_containing(detections)
            active_ids = set(int(t) for t in (detections.tracker_id if detections.tracker_id is not None else []))
            self._update_journeys(arm_map, active_ids, frame_idx)

            # --- Record positions ---
            for i, tid in enumerate(detections.tracker_id if detections.tracker_id is not None else []):
                tid = int(tid)
                x1, y1, x2, y2 = detections.xyxy[i]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if tid in self.journeys:
                    self.journeys[tid].positions.append((cx, cy))

            # --- Annotate Frame ---
            frame = self._draw_paths(frame, frame_idx)
            for name, annotator in self.zone_annotators.items():
                frame = annotator.annotate(scene=frame)

            labels = self._build_labels(detections)
            frame  = self.box_annotator.annotate(scene=frame, detections=detections)
            frame  = self.label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )
            # frame  = self._draw_hud(frame)

            writer.write(frame)

            if frame_idx > 0 and frame_idx % fps == 0:
                duration = fps / (time.time() - start_time)
                start_time = time.time()
                print("Frame %d, t=%.1f d=%.1f" % (frame_idx, frame_idx / fps, duration))

            frame_idx += 1

        cap.release()
        writer.release()
        self._print_summary()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Zone Setup Helper  (run this first to find your polygon coordinates)
# ---------------------------------------------------------------------------

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
    print("\nCopy the above into ARM_POLYGONS in roundabout_tracker.py")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roundabout vehicle maneuver tracker")
    parser.add_argument("--source",  default=SOURCE_VIDEO, help="Input video path")
    parser.add_argument("--output",  default=OUTPUT_VIDEO, help="Output video path")
    parser.add_argument("--setup-zones", action="store_true",
                        help="Run interactive zone setup tool instead of tracking")
    args = parser.parse_args()

    if args.setup_zones:
        setup_zones_interactively(args.source)
    else:
        tracker = RoundaboutTracker(source=args.source, output=args.output)
        tracker.run()
