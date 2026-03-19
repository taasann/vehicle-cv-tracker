# vehicle-cv-tracker

Tracks vehicles in drone footage of a roundabout using YOLO detection and ByteTrack, recording which arm each vehicle enters and exits from.

## Files

- `tracker.py` — main entry point; `VehicleTracker` class, `load_zones()`, all detection/tracking/annotation logic
- `setup_zones.py` — interactive tool to define arm polygons by clicking on a video frame; writes `project.json`
- `project.json` — per-footage zone configuration (arm name → polygon coordinates); loaded at runtime by `tracker.py`

## Setup

```
pip install ultralytics supervision opencv-python numpy
```

Requires a YOLO model at `yolo26x.pt` and input footage at `footage.mp4` (paths configurable via CLI args).

## Workflow

1. Define zones for your footage: `python setup_zones.py --video footage.mp4`
2. Run the tracker: `python tracker.py --source footage.mp4 --output output.mp4`

Both scripts accept `--project` to use a non-default JSON path.

## Key constants (`tracker.py`)

- `VEHICLE_CLASS_IDS` — COCO class IDs detected (car, motorcycle, bus, truck)
- `CONFIDENCE_THRESHOLD` — detection confidence cutoff (default 0.05)
- `TRACK_TIMEOUT_FRAMES` — frames before an unseen journey is dropped (default 120)
- `ZONE_COLORS` — hex colors assigned to arms in order; extend if more than 5 arms

## project.json format

```json
{
  "zones": {
    "ArmName": {
      "polygon": [[x, y], [x, y], ...],
      "color": "#E41A1C"
    }
  }
}
```

Polygons should cover both entry and exit lanes of each arm. Arm names are arbitrary strings. Colors are hex strings; `setup_zones.py` assigns them automatically from a 10-color palette.
