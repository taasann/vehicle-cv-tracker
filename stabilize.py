"""
Video stabilization using point feature tracking (Lucas-Kanade optical flow).

Tracks sparse feature points across frames to estimate the camera motion,
then applies a smoothed inverse transform to remove shake.
"""

import argparse
import sys

import cv2
import numpy as np


def moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    kernel_size = 2 * radius + 1
    kernel = np.ones(kernel_size) / kernel_size
    padded = np.pad(curve, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_trajectory(trajectory: np.ndarray, radius: int) -> np.ndarray:
    smoothed = np.copy(trajectory)
    for col in range(trajectory.shape[1]):
        smoothed[:, col] = moving_average(trajectory[:, col], radius)
    return smoothed


def estimate_transform(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Return a 2x3 affine matrix from prev_gray to curr_gray, or identity on failure."""
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=30,
        blockSize=3,
    )
    if prev_pts is None:
        return np.eye(2, 3, dtype=np.float64)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if curr_pts is None or status is None:
        return np.eye(2, 3, dtype=np.float64)

    mask = status.ravel() == 1
    if mask.sum() < 6:
        return np.eye(2, 3, dtype=np.float64)

    matrix, _ = cv2.estimateAffinePartial2D(
        prev_pts[mask], curr_pts[mask], method=cv2.RANSAC
    )
    if matrix is None:
        return np.eye(2, 3, dtype=np.float64)

    return matrix


def decompose_affine(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (dx, dy, rotation_radians) from a 2x3 affine matrix."""
    dx = matrix[0, 2]
    dy = matrix[1, 2]
    angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    return dx, dy, angle


def stabilize(input_path: str, output_path: str, smoothing_radius: int) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Error: cannot open '{input_path}'")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_str = "mp4v"

    print(f"Input:  {input_path}  ({width}x{height}, {fps:.2f} fps, {n_frames} frames)")
    print(f"Output: {output_path}  smoothing_radius={smoothing_radius}")

    # --- Pass 1: accumulate transforms ---
    transforms = np.zeros((n_frames - 1, 3), dtype=np.float64)  # dx, dy, angle

    ret, prev_frame = cap.read()
    if not ret:
        sys.exit("Error: could not read first frame")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i in range(n_frames - 1):
        ret, curr_frame = cap.read()
        if not ret:
            n_frames = i + 1
            transforms = transforms[:i]
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        matrix = estimate_transform(prev_gray, curr_gray)
        transforms[i] = decompose_affine(matrix)
        prev_gray = curr_gray

        if (i + 1) % 100 == 0:
            print(f"  Pass 1: {i + 1}/{n_frames - 1} frames analysed")

    cap.release()

    # Cumulative sum → trajectory, smooth it, compute correction
    trajectory = np.cumsum(transforms, axis=0)
    smoothed = smooth_trajectory(trajectory, smoothing_radius)
    corrections = smoothed - trajectory  # shape (n_frames-1, 3)

    # --- Pass 2: apply corrections ---
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, frame = cap.read()
    out.write(frame)  # first frame is unchanged

    for i in range(len(corrections)):
        ret, frame = cap.read()
        if not ret:
            break

        dx, dy, angle = corrections[i]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        stabilization_matrix = np.array(
            [[cos_a, -sin_a, dx],
             [sin_a,  cos_a, dy]],
            dtype=np.float64,
        )
        stabilized = cv2.warpAffine(frame, stabilization_matrix, (width, height))
        out.write(stabilized)

        if (i + 1) % 100 == 0:
            print(f"  Pass 2: {i + 1}/{len(corrections)} frames written")

    cap.release()
    out.release()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stabilize drone footage using Lucas-Kanade optical flow."
    )
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("output", help="Path for stabilized output video")
    parser.add_argument(
        "--smoothing-radius",
        type=int,
        default=50,
        help="Moving-average radius in frames (default: 50). "
             "Higher = smoother but more crop/border.",
    )
    args = parser.parse_args()

    stabilize(args.input, args.output, args.smoothing_radius)


if __name__ == "__main__":
    main()
