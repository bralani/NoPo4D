"""Base dataset class and shared loading utilities."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Int64
from torch import Tensor
from torch.utils.data import Dataset

from .types import CameraFrames, FrameData, SceneCameras, SceneFrame, Stage, UnbatchedExample
from .view_sampler import ViewSampler, ViewSamplerCfg
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from src.utils.geometry import get_fov


@dataclass
class DatasetCfgCommon:
    name: str                         # dataset name
    input_image_shape: list[int]      # [H, W]
    view_sampler: ViewSamplerCfg      # config for how to sample context/target views from each scene
    max_fov: float | None = None      # reject scenes whose FOV exceeds this (degrees)
    augment: bool = False             # enable image augmentation at train time
    intr_augment: bool = False        # enable random-scale intrinsic augmentation at train time
    max_retry_per_item: int = 20      # retries with a random scene on failed __getitem__


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets.

    To implement a new dataset, subclass this and:
    1. Implement `load_scene_metadata(scene_path)` — called in parallel for each scene path.
       Return (cam_frames, scene_id, cameras) where cam_frames is CameraFrames.
    2. Implement `load_frames(frames, scene)` — called per batch item to load image data.
    3. In `__init__`, call `super().__init__(...)` then `self._load_scenes(scene_paths)`.
    """

    cfg: DatasetCfgCommon                       # dataset config
    stage: Stage                                # "train" or "val"
    view_sampler: ViewSampler                   # selects which frames to use as context / target
    scene_ids: list[str]                        # ordered scene ids
    scene_to_cameras: dict[str, SceneCameras]   # maps scene_id to cameras used in that scene
    scene_to_frames: dict[str, CameraFrames]    # maps scene_id to {cam_id -> frames} for that scene

    def __init__(self, cfg: DatasetCfgCommon, stage: Stage, view_sampler: ViewSampler) -> None:
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.scene_ids = []
        self.scene_to_cameras = {}
        self.scene_to_frames = {}

    @abstractmethod
    def load_scene_metadata(self, scene_path: str | Path) -> tuple[CameraFrames, str, SceneCameras]:
        """Load all metadata for one scene directory.

        Returns (cam_frames, scene_id, cameras):
          - cam_frames: {cam_id -> [SceneFrame, ...]} for every camera in the scene
          - scene_id:   unique string key used in all lookup dicts
          - cameras:    per-camera intrinsics and extrinsics for every camera in the scene
        """
        ...

    @abstractmethod
    def load_frames(self, frames: list[SceneFrame], scene: str) -> FrameData:
        """Load actual image data for the given frame list.

        Returns FrameData with stacked `images`, `extrinsics`, `intrinsics` in the same order as `frames`.
        """
        ...

    def _load_scenes(self, scene_paths: list[str]) -> None:
        """Parallel metadata loading. Populates scene_to_frames, scene_ids, scene_to_cameras."""
        with ThreadPoolExecutor(max_workers=min(32, len(scene_paths) or 1)) as executor:
            futures = {executor.submit(self.load_scene_metadata, p): p for p in scene_paths}
            for f in as_completed(futures):
                try:
                    frames, s_id, cams = f.result()
                    self.scene_to_frames[s_id]  = frames
                    self.scene_to_cameras[s_id] = cams
                    self.scene_ids.append(s_id)
                except Exception as e:
                    print(f"Warning: failed to load {futures[f]}: {e}")

    def get_frames_at_timesteps(
        self,
        scene: str,
        time_indices: Int64[Tensor, "num_times"],
        selected_cameras: list[str],
    ) -> list[SceneFrame]:
        """Return frames for the given cameras at the given time indices, in camera-major order."""
        frames = self.scene_to_frames[scene]
        return [
            frames[cam][int(t)]
            for cam in selected_cameras
            for t in time_indices
            if int(t) < len(frames.get(cam, []))
        ]

    def get_frames_for_sampler(self, scene: str, num_cameras: int = 1) -> tuple[list[SceneFrame], list[str]]:
        """Randomly select cameras, validate FOV, and return first-camera frames with the selected camera ids."""
        cam_ids = list(self.scene_to_cameras[scene].keys())
        n = min(num_cameras, len(cam_ids))
        selected_cameras = np.random.choice(cam_ids, size=n, replace=False).tolist()
        first_camera = selected_cameras[0]

        frames = self.scene_to_frames[scene][first_camera]

        if self.cfg.max_fov is not None:
            intr = self.scene_to_cameras[scene][first_camera]["intrinsics"].unsqueeze(0)
            fov  = get_fov(intr).rad2deg()
            if (fov > float(self.cfg.max_fov)).any():
                raise ValueError(f"Field of view too wide: {fov}. Max: {self.cfg.max_fov}")

        return frames, selected_cameras

    def add_timestamps(
        self,
        example: UnbatchedExample,
        frames: list[SceneFrame],
        context_indices: Int64[Tensor, "context_views"],
        target_indices: Int64[Tensor, "target_views"],
        num_selected_cameras: int,
    ) -> UnbatchedExample:
        """Add normalized timestamps, tiled by num_cameras for camera-major layouts."""
        if "num_frame" not in frames[0]:
            return example

        context_ts = [frames[i]["num_frame"] for i in context_indices]
        target_ts  = [frames[i]["num_frame"] for i in target_indices]
        context_ts = torch.tensor(context_ts, dtype=torch.float32).repeat(num_selected_cameras)
        target_ts  = torch.tensor(target_ts,  dtype=torch.float32).repeat(num_selected_cameras)

        t_min   = context_ts.min()
        t_range = (context_ts.max() - t_min).clamp_min(1e-8)

        example["context"]["timestamp"] = (context_ts - t_min) / t_range
        example["target"]["timestamp"]  = (target_ts  - t_min) / t_range
        return example

    def _getitem(
        self,
        index: int,
        num_context: int,
        image_shape: tuple[int, int],
        num_target: int = 0,
        num_cameras: int = 1,
    ) -> UnbatchedExample:
        """Sample one training example: select cameras, sample frame indices, load images, apply augmentation."""
        scene = self.scene_ids[index]
        frames, selected_cameras = self.get_frames_for_sampler(scene, num_cameras)

        context_indices, target_indices = self.view_sampler.sample(
            scene=scene,
            num_context_views=num_context,
            num_frames=len(frames),
            num_target_views=num_target,
        )

        context_frames = self.get_frames_at_timesteps(scene, context_indices, selected_cameras)
        target_frames  = self.get_frames_at_timesteps(scene, target_indices,  selected_cameras)
        n_context      = len(context_frames)

        data = self.load_frames(context_frames + target_frames, scene)

        example: UnbatchedExample = {
            "context":     {"extrinsics": data["extrinsics"][:n_context], "intrinsics": data["intrinsics"][:n_context], "image": data["images"][:n_context]},
            "target":      {"extrinsics": data["extrinsics"][n_context:], "intrinsics": data["intrinsics"][n_context:], "image": data["images"][n_context:]},
            "scene":       f"{self.cfg.name}_{scene}",
            "num_cameras": len(selected_cameras),
        }

        example = self.add_timestamps(example, frames, context_indices, target_indices, len(selected_cameras))

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        example = apply_crop_shim(
            example=example,
            shape=image_shape,
            intr_aug=(self.stage == "train" and self.cfg.intr_augment),
        )
        return example

    def __getitem__(self, index_tuple: tuple[int, ...]) -> UnbatchedExample:
        idx, num_context, num_target, num_cameras = index_tuple
        shape = tuple(self.cfg.input_image_shape)

        for _ in range(self.cfg.max_retry_per_item):
            try:
                return self._getitem(idx, num_context, shape, num_target, num_cameras)
            except Exception as e:
                last_err = e
                idx = np.random.randint(len(self))
        raise RuntimeError(f"Failed after {self.cfg.max_retry_per_item} retries. Last error: {last_err}")

    def __len__(self) -> int:
        return len(self.scene_ids)
