"""Exo4D dataset for synchronized multi-camera exocentric video scenes."""

import csv
import hashlib
import os
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from jaxtyping import Float, Int64
from PIL import Image
from torch import Tensor

from .dataset import BaseDataset, DatasetCfgCommon
from .types import CameraCalibration, FrameData, SceneFrame, Stage, UnbatchedExample
from .utils import convert_intrinsics
from .view_sampler import ViewSampler
from src.utils.geometry import quat_to_mat

CACHE_VERSION = 12


@dataclass
class DatasetExo4DCfg(DatasetCfgCommon):
    """Configuration for Exo4D dataset."""
    roots: list[Path]
    annotations_root: Path
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    relative_pose: bool
    skip_bad_shape: bool
    augment: bool
    intr_augment: bool
    normalize_by_pts3d: bool
    rescale_to_1cube: bool


@dataclass
class DatasetExo4DCfgWrapper:
    exo4d: DatasetExo4DCfg


class DatasetExo4D(BaseDataset):
    """Exo4D dataset class for loading and processing exocentric camera scenes."""
    cfg: DatasetExo4DCfg

    # Additional data structures for multi-camera synchronized access
    frames_by_camera_and_time: dict[str, dict[str, list[SceneFrame]]]
    camera_ids: dict[str, list[str]]
    num_time_steps: dict[str, int]
    _h5_cam_order: dict[str, list[str]]  # original cam order in HDF5 file per scene

    def __init__(
        self,
        cfg: DatasetExo4DCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        """Initialize dataset and preload scene metadata."""
        super().__init__(cfg, stage, view_sampler)

        # Initialize multi-camera data structures
        self.frames_by_camera_and_time = {}
        self.camera_ids = {}
        self.num_time_steps = {}
        self._h5_cam_order = {}

        # Resolve paths: data_root for images, annotations_root for JSON metadata
        self.data_root = Path(cfg.roots[0])
        self.annotations_root = Path(cfg.annotations_root)

        index_path = self.data_root / f"sequences_{self.data_stage}.txt"
        with index_path.open("r") as fh:
            index_content = fh.read()
        data_index = [
            line.strip()
            for line in index_content.splitlines()
            if line.strip()
        ]
        filtered_index_content = "\n".join(data_index)

        cache_path = self._get_cache_path(filtered_index_content)
        if not self._try_load_cache(cache_path):
            scene_paths = [str(self.data_root / item) for item in data_index]
            self._load_scenes(scene_paths)
            self._save_cache(cache_path)

        print(f"Exo4D: {self.stage}: loaded {len(self.index_to_scene)} scenes")

    def _get_cache_path(self, index_content: str) -> Path:
        content_hash = hashlib.md5(index_content.encode()).hexdigest()[:12]
        cache_dir = self.data_root / ".cache"
        return cache_dir / f"metadata_{self.data_stage}_v{CACHE_VERSION}_{content_hash}.pkl"

    def _try_load_cache(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.scenes_by_id = data["scenes_by_id"]
            self.index_to_scene = data["index_to_scene"]
            self.frames_by_camera_and_time = data["frames_by_camera_and_time"]
            self.camera_ids = data["camera_ids"]
            self.num_time_steps = data["num_time_steps"]
            self.camera_calibrations = data["camera_calibrations"]
            self._h5_cam_order = data.get("h5_cam_order", {})
            return True
        except Exception as e:
            print(f"Warning: failed to load metadata cache from {cache_path}: {e}. Rebuilding.")
            return False

    def _save_cache(self, cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scenes_by_id": self.scenes_by_id,
            "index_to_scene": self.index_to_scene,
            "frames_by_camera_and_time": self.frames_by_camera_and_time,
            "camera_ids": self.camera_ids,
            "num_time_steps": self.num_time_steps,
            "camera_calibrations": self.camera_calibrations,
            "h5_cam_order": self._h5_cam_order,
        }
        tmp_path = ""
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".pkl.tmp")
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            print(f"Warning: failed to save metadata cache to {cache_path}: {e}")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _load_camera_pose_data(self, take_name: str, take_path: Path) -> dict[str, CameraCalibration] | None:
        """Load camera poses from gopro_calibs.csv and normalize into CameraCalibration tensors.

        Returns:
            Dictionary mapping cam_id -> CameraCalibration, or None if file not found.
        """
        calibs_path = self.annotations_root / "takes" / take_name / "trajectory" / "gopro_calibs.csv"
        if not calibs_path.exists():
            return None

        camera_data: dict[str, CameraCalibration] = {}

        with open(calibs_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cam_id = row["cam_uid"]

                # Parse translation and quaternion (x, y, z, w format)
                tx = float(row["tx_world_cam"])
                ty = float(row["ty_world_cam"])
                tz = float(row["tz_world_cam"])
                qx = float(row["qx_world_cam"])
                qy = float(row["qy_world_cam"])
                qz = float(row["qz_world_cam"])
                qw = float(row["qw_world_cam"])

                # Convert quaternion to rotation matrix
                quat = torch.tensor([qx, qy, qz, qw], dtype=torch.float64)
                rot_mat = quat_to_mat(quat).numpy()

                # Build 4x4 extrinsics matrix
                extrinsics_np = np.eye(4, dtype=np.float64)
                extrinsics_np[:3, :3] = rot_mat
                extrinsics_np[:3, 3] = [tx, ty, tz]

                # Parse intrinsics (fx, fy, cx, cy)
                fx = float(row["intrinsics_0"])
                fy = float(row["intrinsics_1"])
                cx = float(row["intrinsics_2"])
                cy = float(row["intrinsics_3"])

                camera_data[cam_id] = {
                    "intrinsics": convert_intrinsics({
                        "h": 2 * cy, "w": 2 * cx,
                        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    }),
                    "extrinsics": torch.tensor(extrinsics_np, dtype=torch.float32),
                }
        return camera_data

    def load_scene_metadata(
        self, scene_path: str | Path
    ) -> tuple[list[SceneFrame], str, dict[str, CameraCalibration]]:
        """Load metadata for all exocentric cameras in a scene.

        Auto-detects format: if {take_name}.h5 exists, uses HDF5 format;
        otherwise falls back to legacy per-camera PNG folder format.

        Returns:
            Tuple of (scene_frames, scene_id, calibrations). Each frame includes camera_id.
        """
        scene_path = Path(scene_path)
        take_name = scene_path.name
        scene_id = take_name
        if (scene_path / f"{take_name}.h5").exists():
            return self._load_scene_metadata_hdf5(scene_path, take_name, scene_id)
        return self._load_scene_metadata_png(scene_path, take_name, scene_id)

    def _load_scene_metadata_png(
        self, scene_path: Path, take_name: str, scene_id: str
    ) -> tuple[list[SceneFrame], str, dict[str, CameraCalibration]]:
        """Load metadata from legacy per-camera PNG folder format."""
        # Load camera pose data from gopro_calibs.csv
        camera_pose_data = self._load_camera_pose_data(take_name, scene_path)
        if camera_pose_data is None:
            raise ValueError(f"Could not find gopro_calibs.csv for take '{take_name}' at {scene_path}")

        # Find all camera folders (cam01, cam02, etc.) in processed directory
        cam_folders = sorted([
            d for d in scene_path.iterdir()
            if d.is_dir() and d.name.startswith("cam")
        ])

        if not cam_folders:
            raise ValueError(f"No camera folders found in {scene_path}")

        frames_by_cam: dict[str, list[SceneFrame]] = {}
        scene_calibrations: dict[str, CameraCalibration] = {}

        for cam_folder in cam_folders:
            cam_id = cam_folder.name  # e.g., "cam01"

            # Get list of frame images
            frame_files = sorted(cam_folder.glob("frame_*.png"))
            if not frame_files:
                print(f"Warning: No frames found in {cam_folder}")
                continue

            # Get camera parameters
            calib = camera_pose_data.get(cam_id)
            if calib is None:
                print(f"Warning: No camera parameters found for {cam_id}, skipping")
                continue

            scene_calibrations[cam_id] = calib

            frames_by_cam[cam_id] = []
            for frame_idx, frame_path in enumerate(frame_files):
                frame_info: SceneFrame = {
                    "file_path": str(frame_path),
                    "num_frame": frame_idx,
                    "camera_id": cam_id,
                }
                frames_by_cam[cam_id].append(frame_info)

        cam_ids_sorted = sorted(frames_by_cam.keys())
        self.frames_by_camera_and_time[scene_id] = frames_by_cam
        self.camera_ids[scene_id] = cam_ids_sorted
        self.num_time_steps[scene_id] = len(frames_by_cam[cam_ids_sorted[0]]) if cam_ids_sorted else 0
        return frames_by_cam[cam_ids_sorted[0]], scene_id, scene_calibrations

    def _load_scene_metadata_hdf5(
        self, scene_path: Path, take_name: str, scene_id: str
    ) -> tuple[list[SceneFrame], str, dict[str, CameraCalibration]]:
        """Load metadata from HDF5 format ({take_name}.h5 present)."""
        h5_path = str(scene_path / f"{take_name}.h5")
        with h5py.File(h5_path, "r") as hf:
            cam_ids = [
                c.decode() if isinstance(c, bytes) else c
                for c in hf["camera_ids"][:]
            ]
            num_frames = int(hf.attrs["num_frames"])
            intrinsics_arr = hf["intrinsics"][:]   # [num_cameras, 3, 3]
            extrinsics_arr = hf["extrinsics"][:]   # [num_cameras, 4, 4]
            w = int(hf.attrs["width"])
            h = int(hf.attrs["height"])

        cam_ids_sorted = sorted(cam_ids)
        cam_to_idx = {c: i for i, c in enumerate(cam_ids)}  # original file order

        # Build calibration registry (pixel-space K -> normalized)
        scene_calibrations: dict[str, CameraCalibration] = {}
        for cam_id in cam_ids_sorted:
            idx = cam_to_idx[cam_id]
            K_raw = intrinsics_arr[idx]   # 3x3 in pixel space
            meta = {
                "fx": float(K_raw[0, 0]), "fy": float(K_raw[1, 1]),
                "cx": float(K_raw[0, 2]), "cy": float(K_raw[1, 2]),
                "w": w, "h": h,
            }
            scene_calibrations[cam_id] = {
                "intrinsics": convert_intrinsics(meta),
                "extrinsics": torch.tensor(extrinsics_arr[idx], dtype=torch.float32),
            }

        # Build SceneFrame list per camera (lightweight: indices + path)
        frames_by_cam: dict[str, list[SceneFrame]] = {c: [] for c in cam_ids_sorted}
        for cam_id in cam_ids_sorted:
            for frame_idx in range(num_frames):
                frames_by_cam[cam_id].append({
                    "h5_path": h5_path,
                    "num_frame": frame_idx,
                    "camera_id": cam_id,
                })

        self.frames_by_camera_and_time[scene_id] = frames_by_cam
        self.camera_ids[scene_id] = cam_ids_sorted
        self.num_time_steps[scene_id] = num_frames
        self._h5_cam_order[scene_id] = cam_ids   # original order in file
        return frames_by_cam[cam_ids_sorted[0]], scene_id, scene_calibrations

    def get_num_cameras(self, scene: str) -> int:
        """Get the number of cameras in a scene."""
        return len(self.camera_ids.get(scene, []))

    def select_cameras_for_scene(self, scene: str) -> list[str]:
        """Select cameras to use for this scene (called before sampling).
        
        Randomly selects 2 to N cameras and stores the selection for later use.
        For exocentric data, we need at least 2 cameras for meaningful training.
        Returns the selected camera IDs.
        """
        cam_ids = self.camera_ids.get(scene, [])
        if not cam_ids:
            raise ValueError(f"No cameras found for scene {scene}")
        
        # Use the camera count requested by the batch sampler (synchronized across all GPUs).
        # Falls back to 4 (original default) if not set.
        requested = getattr(self, '_requested_num_cameras', None)
        num_cameras_to_use = int(requested) if requested is not None else 4
        # Clamp to the number of available cameras in this scene.
        num_cameras_to_use = min(num_cameras_to_use, len(cam_ids))
        cam_ids_selected = np.random.choice(
            cam_ids, size=num_cameras_to_use, replace=False
        ).tolist()
        
        # Store selection for use by get_frames_for_indices
        self._selected_cameras = cam_ids_selected
        self._num_cameras_used = num_cameras_to_use
        
        return cam_ids_selected

    def get_frames_for_sampler(self, scene: str) -> tuple[list[SceneFrame], Tensor, Tensor]:
        """Get frames from first camera only for view sampler.

        Also selects which cameras will be used for this scene.
        """
        frames_by_cam = self.frames_by_camera_and_time.get(scene, {})

        # Select cameras for this scene (sets self._selected_cameras and self._num_cameras_used)
        cam_ids_selected = self.select_cameras_for_scene(scene)

        # Use first selected camera's frames for sampling
        first_cam = cam_ids_selected[0]
        first_cam_frames = frames_by_cam.get(first_cam, [])

        # Look up calibration from registry (same values for all frames of static cameras)
        calib = self.camera_calibrations[scene][first_cam]
        n = len(first_cam_frames)
        extrinsics = calib["extrinsics"].unsqueeze(0).expand(n, -1, -1)
        intrinsics = calib["intrinsics"].unsqueeze(0).expand(n, -1, -1)

        return first_cam_frames, extrinsics, intrinsics

    def get_frames_for_indices(
        self,
        scene: str,
        time_indices: Int64[Tensor, "num_times"],
        is_context: bool = True,
    ) -> tuple[list[SceneFrame], Int64[Tensor, "num_views"]]:
        """Get frames and expanded indices for multi-camera setup.
        
        Uses cameras pre-selected by select_cameras_for_scene() (called via get_frames_for_sampler).
        Layout is camera-major: [cam0_t0, cam0_t1, ..., cam1_t0, cam1_t1, ...].
        Indices reflect actual positions in the flattened frames list.
        """
        frames_by_cam: dict[str, list[SceneFrame]] = self.frames_by_camera_and_time.get(scene, {})
        
        # Use pre-selected cameras (set by select_cameras_for_scene via get_frames_for_sampler)
        cam_ids_selected = getattr(self, "_selected_cameras", None)
        if cam_ids_selected is None:
            raise ValueError("Cameras not selected. Call get_frames_for_sampler first.")

        num_cameras = len(cam_ids_selected)
        num_frames_per_camera = self.num_time_steps.get(scene, 0)

        # Build list of frames: camera-major order [cam0_t0, cam0_t1, ..., cam1_t0, ...]
        all_frames: list[SceneFrame] = []
        expanded_indices_list: list[Tensor] = []
        
        for cam_idx, cam_id in enumerate(cam_ids_selected):
            cam_frames = frames_by_cam.get(cam_id, [])
            cam_offset = cam_idx * num_frames_per_camera
            
            for t_idx in time_indices:
                t_idx_int = int(t_idx)
                if t_idx_int < len(cam_frames):
                    all_frames.append(cam_frames[t_idx_int])
            
            # Add offset to time indices for this camera
            expanded_indices_list.append(time_indices + cam_offset)

        expanded_indices = torch.cat(expanded_indices_list)

        return all_frames, expanded_indices

    def load_frames(
        self, frames: list[SceneFrame], scene: str, max_workers: int = 8
    ) -> FrameData:
        """Load images (and optional precomputed tracks) for the given frames.

        Dispatches to _load_frames_hdf5() if frames use HDF5 format
        (detected by presence of "h5_path"), otherwise to _load_frames_png().

        Returns:
            FrameData with images, extrinsics, intrinsics and optionally
            tracks and visibility.
        """
        if not frames:
            raise ValueError("No frames provided to load_frames")
        calib = self.camera_calibrations[scene]
        if "h5_path" in frames[0]:
            return self._load_frames_hdf5(frames, calib, scene)
        return self._load_frames_png(frames, calib, max_workers)

    def _load_frames_png(
        self,
        frames: list[SceneFrame],
        calib: dict[str, CameraCalibration],
        max_workers: int = 8,
    ) -> FrameData:
        """Load frames from legacy per-camera PNG files."""
        def load_single(frame: SceneFrame):
            img = self.to_tensor(Image.open(frame["file_path"]).convert("RGB"))
            frame_path = Path(frame["file_path"])
            cam_id = frame_path.parent.name          # e.g. "cam01"
            scene_root = frame_path.parent.parent    # e.g. .../scene_name
            frame_num = frame_path.stem.split("_")[1]  # "frame_000042" -> "000042"
            track_dir = scene_root / "tracks" / cam_id
            track_path = track_dir / f"track_{frame_num}.pt"
            vis_path   = track_dir / f"visibility_{frame_num}.pt"
            track, visibility = None, None
            if track_path.exists() and vis_path.exists():
                track      = torch.load(track_path, weights_only=True)
                visibility = torch.load(vis_path,   weights_only=True)
            return img, track, visibility

        num_frames = len(frames)
        if num_frames <= 4:
            results = [load_single(f) for f in frames]
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, num_frames)) as executor:
                results = list(executor.map(load_single, frames))

        return self._assemble_frame_data(frames, results, calib)

    def _load_frames_hdf5(
        self,
        frames: list[SceneFrame],
        calib: dict[str, CameraCalibration],
        scene: str,
    ) -> FrameData:
        """Load frames from HDF5 file.

        Reads one chunk per unique timestep (chunk = [1, all_cams, H, W, 3]),
        then assembles the requested (timestep, camera) pairs.
        """
        h5_path = frames[0]["h5_path"]
        cam_ids_file = self._h5_cam_order.get(scene, [])
        cam_to_file_idx = {c: i for i, c in enumerate(cam_ids_file)}

        # Group by timestep to read each HDF5 chunk only once
        timesteps = sorted({f["num_frame"] for f in frames})
        timestep_to_data = {}
        with h5py.File(h5_path, "r") as hf:
            frames_ds = hf["frames"]   # [T, num_cameras, H, W, 3]
            for t in timesteps:
                timestep_to_data[t] = frames_ds[t]

        # Assemble images in the same order as frames list
        raw_img = np.stack([
            timestep_to_data[frame["num_frame"]][cam_to_file_idx[frame["camera_id"]]]
            for frame in frames
        ])
        stacked_images = torch.from_numpy(raw_img).permute(0, 3, 1, 2).float()

        return FrameData(
            images=stacked_images,
            extrinsics=torch.stack([calib[f["camera_id"]]["extrinsics"] for f in frames]),
            intrinsics=torch.stack([calib[f["camera_id"]]["intrinsics"] for f in frames]),
        )

    def _assemble_frame_data(
        self,
        frames: list[SceneFrame],
        results: list[tuple],
        calib: dict[str, CameraCalibration],
    ) -> FrameData:
        """Build a FrameData dict from a list of (img, track, visibility) tuples."""
        images       = [r[0] for r in results]
        tracks       = [r[1] for r in results]
        visibilities = [r[2] for r in results]

        num_frames = len(frames)
        stacked_images = torch.stack(images)
        extrinsics = torch.stack([calib[f["camera_id"]]["extrinsics"] for f in frames])
        intrinsics = torch.stack([calib[f["camera_id"]]["intrinsics"] for f in frames])

        frame_data: FrameData = {
            "images":     stacked_images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
        }

        if not all(t is None for t in tracks):
            valid = [i for i, t in enumerate(tracks) if t is not None]
            if len(valid) == num_frames:
                frame_data["tracks"]     = torch.stack(tracks)        # type: ignore[arg-type]
                frame_data["visibility"] = torch.stack(visibilities)  # type: ignore[arg-type]
            else:
                print(f"[WARNING] Only {len(valid)}/{num_frames} frames have tracks — skipping.")

        return frame_data

    def add_timestamps(
        self,
        example: UnbatchedExample,
        frames: list[SceneFrame],
        context_indices: Int64[Tensor, "n_ctx"],
        target_indices: Int64[Tensor, "n_tgt"],
    ) -> UnbatchedExample:
        """Add normalized timestamps for multi-camera setup.

        Timestamps are expanded to match the multi-camera frame expansion.
        Uses the same number of cameras for both context and target.
        Order: camera-major [cam0_t0, cam0_t1, ..., cam1_t0, cam1_t1, ...]."""

        if "num_frame" not in frames[0]:
            return example

        num_cameras = getattr(self, "_num_cameras_used", 1)

        # Get base timestamps from sampled time indices
        context_timestamps_base = torch.tensor(
            [frames[i].get("num_frame", 0) for i in context_indices], dtype=torch.float32
        )
        target_timestamps_base = torch.tensor(
            [frames[i].get("num_frame", 0) for i in target_indices], dtype=torch.float32
        )

        # Expand timestamps: repeat for each camera in camera-major order
        context_timestamps = context_timestamps_base.repeat(num_cameras)
        target_timestamps = target_timestamps_base.repeat(num_cameras)

        # Normalize by context min-max
        context_min = context_timestamps_base.min()
        context_max = context_timestamps_base.max()
        context_range = context_max - context_min

        epsilon = 1e-8
        if abs(float(context_range)) > epsilon:
            example["context"]["timestamp"] = (context_timestamps - context_min) / context_range
            example["target"]["timestamp"] = (target_timestamps - context_min) / context_range
        else:
            example["context"]["timestamp"] = torch.zeros_like(context_timestamps)
            example["target"]["timestamp"] = torch.zeros_like(target_timestamps)

        return example
