"""Utility functions for EgoExo4D preprocessing.

Covers calibration loading, intrinsics scaling, and writing undistorted
frames from multiple cameras into a single HDF5 file.
"""

import csv
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)  # ensure unit quaternion
    x, y, z, w = q

    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [    2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def load_calibration(trajectory_dir: Path) -> dict | None:
    """Load camera calibration from gopro_calibs.csv.

    Returns a dict mapping each camera ID to its intrinsic matrix K,
    fisheye distortion coefficients, and camera-to-world extrinsic matrix.
    Returns None if the file does not exist.
    """
    csv_path = trajectory_dir / "gopro_calibs.csv"
    if not csv_path.exists():
        return None

    cameras = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cam_id = row["cam_uid"]

            # Intrinsic matrix — gopro_calibs stores [fx, fy, cx, cy] in intrinsics_0..3
            fx, fy, cx, cy = (float(row[f"intrinsics_{i}"]) for i in range(4))
            K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1],
            ], dtype=np.float64)

            # Fisheye distortion coefficients — intrinsics_4..7
            dist = np.array([float(row[f"intrinsics_{i}"]) for i in range(4, 8)], dtype=np.float64)

            # Camera-to-world pose assembled from quaternion rotation and translation
            rotation = quat_to_rotmat(
                float(row["qx_world_cam"]),
                float(row["qy_world_cam"]),
                float(row["qz_world_cam"]),
                float(row["qw_world_cam"]),
            )
            translation = np.array([
                float(row["tx_world_cam"]),
                float(row["ty_world_cam"]),
                float(row["tz_world_cam"]),
            ])
            extrinsic = np.eye(4, dtype=np.float64)
            extrinsic[:3, :3] = rotation
            extrinsic[:3,  3] = translation

            cameras[cam_id] = {"K": K, "dist": dist, "ext": extrinsic}

    return cameras if cameras else None


def scale_intrinsics(K: np.ndarray, video_width: int, video_height: int) -> np.ndarray:
    """Rescale an intrinsic matrix to match the actual video resolution.

    gopro_calibs.csv stores calibration values relative to the full sensor
    resolution (cx = width/2, cy = height/2). This rescales K so that fx, cx
    and fy, cy match the downscaled video dimensions.
    """
    K = K.copy()
    scale_x = video_width  / (2 * K[0, 2])   # 2*cx == full calibration width
    scale_y = video_height / (2 * K[1, 2])   # 2*cy == full calibration height
    K[0, [0, 2]] *= scale_x   # scale fx and cx
    K[1, [1, 2]] *= scale_y   # scale fy and cy
    return K


def read_and_undistort_frame(
    cam_ids: list[str],
    captures: dict[str, cv2.VideoCapture],
    undistortion_maps: dict[str, tuple],
    frame_height: int,
    frame_width: int,
    frame_idx: int,
) -> np.ndarray:
    """Read one frame from each camera, undistort it, and return the stacked chunk.

    Returns an array of shape [N, H, W, 3] in float16 RGB, normalised to [0, 1].
    """
    frame_chunk = np.empty((len(cam_ids), frame_height, frame_width, 3), dtype=np.float16)

    for cam_idx, cam_id in enumerate(cam_ids):
        ok, frame_bgr = captures[cam_id].read()
        if not ok:
            raise RuntimeError(f"Unexpected end of video for camera '{cam_id}' at frame {frame_idx}")
        map_x, map_y = undistortion_maps[cam_id]
        # Undistort fisheye, convert BGR → RGB, normalise to [0, 1]
        undistorted       = cv2.remap(frame_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        frame_rgb         = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        frame_chunk[cam_idx] = frame_rgb.astype(np.float16) / 255.0

    return frame_chunk


def write_hdf5(
    video_paths: dict[str, Path],
    calib: dict[str, dict],
    undistortion_maps: dict[str, tuple],
    output_path: Path,
    silent: bool = False,
) -> int:
    """Undistort frames from all cameras and write them into a single HDF5 file.

    The output file contains:
        frames      [T, N, H, W, 3]  float16   undistorted RGB frames
        intrinsics  [N, 3, 3]        float64   camera intrinsic matrices
        extrinsics  [N, 4, 4]        float64   camera-to-world poses
        camera_ids  [N]              str       camera identifiers

    Returns the number of frames written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cam_ids = sorted(video_paths.keys())
    num_cameras = len(cam_ids)

    captures = {cam_id: cv2.VideoCapture(str(video_paths[cam_id])) for cam_id in cam_ids}
    if not all(cap.isOpened() for cap in captures.values()):
        raise RuntimeError("Failed to open one or more video files")

    try:
        # Use the shortest camera as the frame count to stay in sync across cameras
        num_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures.values())

        frame_height, frame_width = next(iter(undistortion_maps.values()))[0].shape[:2]

        with h5py.File(output_path, "w") as hf:
            # Write static metadata
            hf.create_dataset("camera_ids", data=np.array(cam_ids, dtype=object), dtype=h5py.string_dtype())
            hf.create_dataset("intrinsics",  data=np.stack([calib[cam_id]["K"]   for cam_id in cam_ids]), dtype=np.float64)
            hf.create_dataset("extrinsics",  data=np.stack([calib[cam_id]["ext"] for cam_id in cam_ids]), dtype=np.float64)

            # Pre-allocate the frames dataset with LZF compression for fast writes
            frames_dataset = hf.create_dataset(
                "frames",
                shape=(num_frames, num_cameras, frame_height, frame_width, 3),
                dtype=np.float16,
                chunks=(1, num_cameras, frame_height, frame_width, 3),
                compression="lzf",
            )

            for frame_idx in tqdm(range(num_frames), desc="frames", disable=silent):
                frame_chunk = read_and_undistort_frame(
                    cam_ids, captures, undistortion_maps, frame_height, frame_width, frame_idx
                )
                frames_dataset[frame_idx] = frame_chunk

    finally:
        # Always release captures, even if an error occurred mid-processing
        for cap in captures.values():
            cap.release()

    return num_frames
