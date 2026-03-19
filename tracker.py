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
import json
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_VIDEO = "footage.mp4"
DEFAULT_OUTPUT_VIDEO = "output.mp4"
MODEL_PATH   = "yolo26x.pt"
PROJECT_FILE = "project.json"

# COCO class IDs for vehicles (car, motorcycle, bus, truck)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.05

# How many frames a vehicle must be absent before its journey is "closed"
TRACK_TIMEOUT_FRAMES = 120

# Orphan re-linking: max pixel distance to match a new track to a recently-lost journey
ORPHAN_RELINK_DISTANCE = 120
# How long to keep an orphaned journey available for re-linking (frames)
ORPHAN_RELINK_MAX_FRAMES = 60


# ---------------------------------------------------------------------------
# Zone Loading
# ---------------------------------------------------------------------------

def load_zones(path: str) -> dict[str, dict]:
    """Load arm zones from a project.json file.

    Returns a dict mapping arm name to:
      {"entry_polygon": np.ndarray, "exit_polygon": np.ndarray, "color": str}.
    Falls back to "polygon" for backward compatibility with old project files.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"No {path} found — run setup_zones.py first")
        sys.exit(1)
    return {
        name: {
            "entry_polygon": np.array(
                zone.get("entry_polygon") or zone["polygon"], dtype=np.int32
            ),
            "exit_polygon": np.array(
                zone.get("exit_polygon") or zone["polygon"], dtype=np.int32
            ),
            "color": zone["color"],
        }
        for name, zone in data["zones"].items()
    }


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

class VehicleTracker:
    def __init__(self, source: str, output: str, project: str = PROJECT_FILE):
        self.source_path = source
        self.output_path = output

        zone_data = load_zones(project)

        # Detection model
        self.model = YOLO(MODEL_PATH)

        # ByteTrack tracker
        self.tracker = sv.ByteTrack(lost_track_buffer=300)

        # Build PolygonZone objects for entry and exit polygons per arm
        self.entry_zones: dict[str, sv.PolygonZone] = {
            name: sv.PolygonZone(polygon=z["entry_polygon"])
            for name, z in zone_data.items()
        }
        self.exit_zones: dict[str, sv.PolygonZone] = {
            name: sv.PolygonZone(polygon=z["exit_polygon"])
            for name, z in zone_data.items()
        }

        # Per-vehicle journey state
        self.journeys: dict[int, VehicleJourney] = {}

        # Incomplete journeys that timed out, kept briefly for re-linking
        # Each entry: (journey, last_position, frame_when_orphaned)
        self.orphaned_journeys: list[tuple[VehicleJourney, tuple[int, int], int]] = []

        # Annotators for visualisation
        self.box_annotator   = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.entry_zone_annotators = {
            name: sv.PolygonZoneAnnotator(
                zone=self.entry_zones[name],
                color=sv.Color.from_hex(z["color"]),
                thickness=2,
            )
            for name, z in zone_data.items()
        }
        self.exit_zone_annotators = {
            name: sv.PolygonZoneAnnotator(
                zone=self.exit_zones[name],
                color=sv.Color.from_hex(z["color"]),
                thickness=1,
            )
            for name, z in zone_data.items()
        }

        # BGR color per arm for path drawing
        self.arm_colors: dict[str, tuple[int, int, int]] = {
            name: tuple(int(z["color"].lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
            for name, z in zone_data.items()
        }

    # ------------------------------------------------------------------
    # Zone Logic
    # ------------------------------------------------------------------

    def _arms_in_entry(self, detections: sv.Detections) -> dict[int, str]:
        """Return {track_id: arm_name} for vehicles inside an entry polygon."""
        result = {}
        for arm_name, zone in self.entry_zones.items():
            mask = zone.trigger(detections=detections)
            for i, in_zone in enumerate(mask):
                if in_zone and detections.tracker_id is not None:
                    tid = int(detections.tracker_id[i])
                    result[tid] = arm_name
        return result

    def _arms_in_exit(self, detections: sv.Detections) -> dict[int, str]:
        """Return {track_id: arm_name} for vehicles inside an exit polygon."""
        result = {}
        for arm_name, zone in self.exit_zones.items():
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
        entry_map: dict[int, str],
        exit_map: dict[int, str],
        active_ids: set[int],
        frame_idx: int,
    ) -> None:
        """
        Update each vehicle's journey state based on which zones it's in.

        State transitions:
          - Vehicle is in an entry polygon and has no entry arm → record entry arm
          - Vehicle is in an exit polygon AND already has an entry arm → record
            exit arm and finalise maneuver (u-turns allowed: exit may equal entry)
        """
        # Initialise journeys for new track IDs
        for tid in active_ids:
            if tid not in self.journeys:
                self.journeys[tid] = VehicleJourney(track_id=tid)
            self.journeys[tid].last_seen_frame = frame_idx

        for tid, arm in entry_map.items():
            journey = self.journeys.get(tid)
            if journey is None or journey.completed:
                continue
            if journey.entry_arm is None:
                journey.entry_arm = arm

        for tid, arm in exit_map.items():
            journey = self.journeys.get(tid)
            if journey is None or journey.completed:
                continue
            if journey.entry_arm is not None and journey.exit_arm is None:
                journey.exit_arm = arm
                journey.completed = True
                print(
                    f"[Frame {frame_idx}] Vehicle {tid}: "
                    f"{journey.entry_arm} → {journey.exit_arm} "
                )

    def _relink_orphans(self, detections: sv.Detections, frame_idx: int) -> None:
        """Match new track IDs to recently-lost incomplete journeys by proximity.

        A new track that appears close to where an orphaned journey was last seen
        inherits that journey's state (entry arm, path, etc.) rather than starting
        a blank journey.
        """
        if not self.orphaned_journeys:
            return

        # Expire stale orphans first
        self.orphaned_journeys = [
            (j, pos, f) for j, pos, f in self.orphaned_journeys
            if frame_idx - f <= ORPHAN_RELINK_MAX_FRAMES
        ]

        if detections.tracker_id is None:
            return

        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            if tid in self.journeys:
                continue  # already known

            x1, y1, x2, y2 = detections.xyxy[i]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            best_idx, best_dist = -1, ORPHAN_RELINK_DISTANCE
            for oi, (_, last_pos, _) in enumerate(self.orphaned_journeys):
                dist = ((cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = oi

            if best_idx >= 0:
                orphan, _, _ = self.orphaned_journeys.pop(best_idx)
                orphan.track_id = tid
                orphan.last_seen_frame = frame_idx
                self.journeys[tid] = orphan
                print(
                    f"[Frame {frame_idx}] Re-linked track {tid} to orphaned journey "
                    f"(entry={orphan.entry_arm}, dist={best_dist:.0f}px)"
                )

    def _orphan_lost_tracks(self, active_ids: set[int], frame_idx: int) -> None:
        """Immediately orphan incomplete journeys whose track ID is no longer active.

        Called every frame so re-linking can happen as soon as the replacement
        track ID appears, without waiting for TRACK_TIMEOUT_FRAMES.
        """
        lost = [
            tid for tid, journey in self.journeys.items()
            if tid not in active_ids
            and not journey.completed
            and journey.entry_arm is not None
            and journey.positions
        ]
        for tid in lost:
            journey = self.journeys.pop(tid)
            self.orphaned_journeys.append((journey, journey.positions[-1], frame_idx))
            print(f"[Frame {frame_idx}] Orphaned track {tid} (entry={journey.entry_arm})")

    def _cleanup_journeys(self, frame_idx: int) -> None:
        """Remove journeys that have not been seen for TRACK_TIMEOUT_FRAMES frames."""
        timed_out = [
            tid
            for tid, journey in self.journeys.items()
            if frame_idx - journey.last_seen_frame > TRACK_TIMEOUT_FRAMES
        ]
        for tid in timed_out:
            journey = self.journeys.pop(tid)
            if not journey.completed:
                print(f"Deleting journey {tid}, entry={journey.entry_arm} exit={journey.exit_arm} completed={journey.completed}")

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

    def run(self, display: bool = False) -> None:
        cap    = cv2.VideoCapture(self.source_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        writer = None
        if not display:
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
            entry_map  = self._arms_in_entry(detections)
            exit_map   = self._arms_in_exit(detections)
            active_ids = set(int(t) for t in (detections.tracker_id if detections.tracker_id is not None else []))
            self._relink_orphans(detections, frame_idx)
            self._update_journeys(entry_map, exit_map, active_ids, frame_idx)
            self._orphan_lost_tracks(active_ids, frame_idx)
            self._cleanup_journeys(frame_idx)

            # --- Record positions ---
            for i, tid in enumerate(detections.tracker_id if detections.tracker_id is not None else []):
                tid = int(tid)
                x1, y1, x2, y2 = detections.xyxy[i]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if tid in self.journeys:
                    self.journeys[tid].positions.append((cx, cy))

            # --- Annotate Frame ---
            frame = self._draw_paths(frame, frame_idx)
            for name, annotator in self.entry_zone_annotators.items():
                frame = annotator.annotate(scene=frame)
            for name, annotator in self.exit_zone_annotators.items():
                frame = annotator.annotate(scene=frame)

            labels = self._build_labels(detections)
            frame  = self.box_annotator.annotate(scene=frame, detections=detections)
            frame  = self.label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )
            # frame  = self._draw_hud(frame)

            if display:
                cv2.imshow("Roundabout Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                writer.write(frame)

            if frame_idx > 0 and frame_idx % fps == 0:
                duration = fps / (time.time() - start_time)
                start_time = time.time()
                print("Frame %d, t=%.1f d=%.1f" % (frame_idx, frame_idx / fps, duration))

            frame_idx += 1

        cap.release()
        if display:
            cv2.destroyAllWindows()
        else:
            writer.release()
        self._print_summary()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roundabout vehicle maneuver tracker")
    parser.add_argument("--source", default=DEFAULT_SOURCE_VIDEO, help="Input video path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_VIDEO, help="Output video path")
    parser.add_argument("--project", default=PROJECT_FILE, help="Project JSON file with zone definitions")
    parser.add_argument("--display", action="store_true",
                        help="Show output in a window instead of writing to file")
    args = parser.parse_args()

    tracker = VehicleTracker(source=args.source, output=args.output, project=args.project)
    tracker.run(display=args.display)
