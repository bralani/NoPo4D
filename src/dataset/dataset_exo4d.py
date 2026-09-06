"""Exo4D dataset for synchronized multi-camera exocentric video scenes."""

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING

import h5py
import numpy as np
import torch

from .dataset import BaseDataset, DatasetCfgCommon
from .types import CameraFrames, FrameData, SceneCameras, SceneFrame, Stage
from .view_sampler import ViewSampler
from src.utils.geometry import convert_intrinsics

CACHE_VERSION = 20
CACHE_KEYS = ["scene_to_frames", "scene_ids", "camera_ids", "scene_to_cameras", "_h5_cam_order"]


@dataclass
class DatasetExo4DCfg(DatasetCfgCommon):
    root: Path = MISSING


@dataclass
class DatasetExo4DCfgWrapper:
    exo4d: DatasetExo4DCfg


class DatasetExo4D(BaseDataset):
    """Exo4D dataset: synchronized multi-camera exocentric video in HDF5 format."""

    cfg: DatasetExo4DCfg                 # dataset config with Exo4D-specific fields
    camera_ids: dict[str, list[str]]     # scene_id -> sorted cam ids
    _h5_cam_order: dict[str, list[str]]  # scene_id -> cam order as stored in HDF5

    def __init__(self, cfg: DatasetExo4DCfg, stage: Stage, view_sampler: ViewSampler) -> None:
        super().__init__(cfg, stage, view_sampler)
        self.camera_ids    = {}
        self._h5_cam_order = {}
        self.data_root     = Path(cfg.root)

        data_index = [l.strip() for l in (self.data_root / f"sequences_{stage}.txt").read_text().splitlines() if l.strip()]
        self._load_index(data_index, stage)

    def _load_index(self, data_index: list[str], stage: Stage) -> None:
        """Load scene metadata from cache if available, otherwise build and cache it."""
        content_hash = hashlib.md5("\n".join(data_index).encode()).hexdigest()[:12]
        cache_path   = self.data_root / ".cache" / f"metadata_{stage}_v{CACHE_VERSION}_{content_hash}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self.__dict__.update(pickle.load(f))
                print(f"Exo4D {stage}: loaded {len(self.scene_ids)} scenes from cache")
                return
            except Exception as e:
                print(f"Cache load failed: {e}. Rebuilding.")

        self._load_scenes([str(self.data_root / item) for item in data_index])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({k: getattr(self, k) for k in CACHE_KEYS}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: failed to save cache: {e}")
        print(f"Exo4D {stage}: loaded {len(self.scene_ids)} scenes")

    def load_scene_metadata(self, scene_path: str | Path) -> tuple[CameraFrames, str, SceneCameras]:
        """Read camera calibrations and build per-camera frame lists from an HDF5 scene file."""
        take_name = Path(scene_path).name
        h5_path   = str(Path(scene_path) / f"{take_name}.h5")

        with h5py.File(h5_path, "r") as hf:
            cam_ids    = [c.decode() if isinstance(c, bytes) else c for c in hf["camera_ids"][:]]
            num_frames = int(hf.attrs["num_frames"])
            width      = int(hf.attrs["width"])
            height     = int(hf.attrs["height"])
            intr       = hf["intrinsics"][:]
            extr       = hf["extrinsics"][:]

        scene_cameras: SceneCameras = {
            cam_id: {
                "intrinsics": convert_intrinsics(intr[i, 0, 0], intr[i, 1, 1], intr[i, 0, 2], intr[i, 1, 2], width, height),
                "extrinsics": torch.tensor(extr[i], dtype=torch.float32),
            }
            for i, cam_id in enumerate(cam_ids)
        }

        sorted_cam_ids = sorted(cam_ids)
        cam_frames: CameraFrames = {
            cam_id: [{"scene_path": h5_path, "num_frame": t, "camera_id": cam_id, "extrinsics": None} for t in range(num_frames)]
            for cam_id in sorted_cam_ids
        }

        self.camera_ids[take_name]    = sorted_cam_ids
        self._h5_cam_order[take_name] = cam_ids
        return cam_frames, take_name, scene_cameras

    def load_frames(self, frames: list[SceneFrame], scene: str) -> FrameData:
        """Load image data for the given frames from HDF5, stacked with their extrinsics and intrinsics."""
        if not frames:
            raise ValueError("No frames provided")

        scene_path = frames[0]["scene_path"]
        cam_idx   = {cam_id: i for i, cam_id in enumerate(self._h5_cam_order.get(scene, []))}
        timesteps = sorted({f["num_frame"] for f in frames})

        with h5py.File(scene_path, "r") as hf:
            frame_data = {t: hf["frames"][t] for t in timesteps}

        raw     = np.stack([frame_data[f["num_frame"]][cam_idx[f["camera_id"]]] for f in frames])
        cameras = self.scene_to_cameras[scene]
        return {
            "images":     torch.from_numpy(raw).permute(0, 3, 1, 2).float(),
            "extrinsics": torch.stack([cameras[f["camera_id"]]["extrinsics"] for f in frames]),
            "intrinsics": torch.stack([cameras[f["camera_id"]]["intrinsics"] for f in frames]),
        }
