"""Dataset base utilities and abstract dataset class.

This module provides a small shared dataset foundation used by concrete
dataset implementations in the project."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Sequence, Optional, Callable

# Third-party
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as tf
from jaxtyping import Float, Int64
from einops import repeat

# Local
from .types import UnbatchedExample, Stage, SceneFrame, FrameData, CameraCalibration
from .view_sampler import ViewSampler, ViewSamplerCfg
from .utils import make_baseline_one, prepare_pts3d_and_normalize
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from ..utils.pose import camera_normalization
from src.utils.geometry import get_fov


@dataclass
class DatasetCfgCommon:
    """Common configuration shared by all dataset implementations."""
    name: str
    original_image_shape: list[int]  # [H, W]
    input_image_shape: list[int]  # [H, W]
    background_color: list[float]  # [R, G, B]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg
    sampling_weight: float  # weight for multi-dataset sampling


class BaseDataset(Dataset, ABC):
    """Base dataset containing common utilities used by concrete dataset classes."""

    stage: Stage
    view_sampler: ViewSampler
    data_root: Path
    near: float = 0.1
    far: float = 100.0
    to_tensor: tf.ToTensor

    def __init__(self, cfg: DatasetCfgCommon, stage: Stage, view_sampler: ViewSampler):
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor: tf.ToTensor = tf.ToTensor()
        self.scenes_by_id: dict[str, list[SceneFrame]] = {} # maps scene identifier (str) -> list[SceneFrame]
        self.index_to_scene: dict[int, str] = {} # maps integer index -> scene identifier (str)
        self.camera_calibrations: dict[str, dict[str, CameraCalibration]] = {}  # scene_id -> camera_id -> calibration

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    def get_bound(self, bound: str, num_views: int) -> Float[Tensor, "num_views"]:
        """Return a tensor of shape (num_views,) filled with the near/far bound."""
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @abstractmethod
    def load_scene_metadata(
        self, scene_path: str | Path
    ) -> tuple[list[SceneFrame], str, dict[str, CameraCalibration]]:
        """
        Load scene metadata from a path.

        Subclasses must implement this to return a tuple (scene_frames, scene_id, calibrations).

        Args:
            scene_path: Absolute path to the scene folder.

        Returns:
            (scene_frames, scene_id, calibrations) where calibrations maps
            camera_id -> CameraCalibration (intrinsics, and optionally extrinsics for
            static cameras).
        """
        raise NotImplementedError("Subclasses must implement load_scene_metadata(scene_path)")

    def _load_scenes(self, scene_paths: list[str]) -> None:
        """Load scenes in parallel and populate scenes_by_id and index_to_scene."""
        max_workers: int = min(32, max(1, len(scene_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.load_scene_metadata, scene_path): scene_path
                for scene_path in scene_paths
            }

            for future in as_completed(future_to_path):
                scene_path = future_to_path[future]
                try:
                    scene_frames, scene_id, calibrations = future.result()
                except Exception as exc:
                    print(f"Warning: failed to load {scene_path}: {exc}")
                    continue

                scene_index = len(self.index_to_scene)
                self.scenes_by_id[scene_id] = scene_frames
                self.index_to_scene[scene_index] = scene_id
                self.camera_calibrations[scene_id] = calibrations

    def get_frames_for_indices(
        self,
        scene: str,
        time_indices: Int64[Tensor, "num_times"],
        is_context: bool = True,
    ) -> tuple[list[SceneFrame], Int64[Tensor, "num_views"]]:
        """Get frames and expanded indices for the given time indices."""
        frames = self.scenes_by_id[scene]
        selected_frames = [frames[i] for i in time_indices]
        # For single-camera datasets, indices are just sequential
        expanded_indices = torch.arange(len(selected_frames), dtype=torch.int64)
        return selected_frames, expanded_indices

    @abstractmethod
    def load_frames(
        self, frames: list[SceneFrame], scene: str, max_workers: int = 32
    ) -> FrameData:
        """Load images for a list of frames with their camera parameters.

        Args:
            frames: List of SceneFrame metadata.
            scene: Scene identifier used to look up calibrations from
                   self.camera_calibrations[scene].
            max_workers: Number of parallel workers for image loading.

        Returns:
            FrameData with images, extrinsics, intrinsics and optionally
            tracks and visibility tensors.
        """
        raise NotImplementedError

    def validate_image_shapes(
        self,
        context_images: Float[Tensor, "N_ctx 3 H W"] | Sequence[Float[Tensor, "3 H W"]],
        target_images: Float[Tensor, "N_tgt 3 H W"] | Sequence[Float[Tensor, "3 H W"]],
        example_key: Optional[str] = None,
    ) -> None:
        """
        Validate that context and target images have the expected shape.

        Args:
            context_images: Tensor or sequence of tensors for context images (N, 3, H, W) or (3, H, W).
            target_images: Tensor or sequence of tensors for target images (N, 3, H, W) or (3, H, W).
            example_key: Optional identifier for logging.

        Raises:
            Exception: If shapes do not match and cfg.skip_bad_shape is True.
        """
        expected = (3, *self.cfg.original_image_shape)

        def _shape_from(obj: Float[Tensor, "N 3 H W"] | Sequence[Float[Tensor, "3 H W"]]) -> tuple | None:
            if isinstance(obj, torch.Tensor):
                return tuple(obj.shape[1:])
            if isinstance(obj, Sequence) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
                return tuple(obj[0].shape)
            return None

        context_shape = _shape_from(context_images)
        # For sequences we expect (C,H,W) stored directly on the per-image tensors;
        # if the returned tuple is length 4 (batched tensor) drop the batch dim.
        if context_shape is not None and len(context_shape) == 4:
            context_shape = context_shape[1:]

        target_shape = _shape_from(target_images)
        if target_shape is not None and len(target_shape) == 4:
            target_shape = target_shape[1:]

        context_invalid = (context_shape != expected) if context_shape is not None else False
        target_invalid = (target_shape != expected) if target_shape is not None else False

        if getattr(self.cfg, "skip_bad_shape", False) and (context_invalid or target_invalid):
            key_str = f" {example_key}" if example_key is not None else ""
            print(
                f"Skipped bad example{key_str}. Context shape was {getattr(context_images, 'shape', context_shape)} "
                f"and target shape was {getattr(target_images, 'shape', target_shape)}."
            )
            raise Exception("Bad example image shape")

    def normalize_extrinsics(
        self,
        extrinsics: Float[Tensor, "num_views 4 4"],
        context_indices: Int64[Tensor, "context_views"],
    ) -> tuple[Float[Tensor, "num_views 4 4"], float]:
        """
        Normalize camera extrinsics using configuration in self.cfg.

        Args:
            extrinsics: Tensor[float32, (num_views, 4, 4)]
            context_indices: Indices of context views.

        Returns:
            normalized_extrinsics: Tensor[float32, (num_views, 4, 4)]
            scale: float
        """
        # Read options from cfg with safe defaults
        make_baseline_1 = getattr(self.cfg, "make_baseline_1", False)
        baseline_min = getattr(self.cfg, "baseline_min", 0.0)
        baseline_max = getattr(self.cfg, "baseline_max", float("inf"))
        relative_pose = getattr(self.cfg, "relative_pose", False)
        rescale_to_1cube = getattr(self.cfg, "rescale_to_1cube", False)

        scale = 1.0
        if make_baseline_1:
            scale = make_baseline_one(
                extrinsics=extrinsics,
                context_indices=context_indices,
                baseline_min=baseline_min,
                baseline_max=baseline_max,
            )

        if relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        if rescale_to_1cube:
            scene_scale = torch.max(torch.abs(extrinsics[context_indices][:, :3, 3]))
            if scene_scale != 0:
                extrinsics[:, :3, 3] /= (1 * scene_scale)

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")

        return extrinsics, scale

    def augment_example(
        self,
        example: UnbatchedExample,
        target_images: Float[Tensor, "N 3 H W"],
        patchsize: tuple[int, int],
    ) -> UnbatchedExample:
        """
        Apply augmentation, cropping and point preparation/normalization to an example.

        Args:
            example: UnbatchedExample dictionary containing 'context' and 'target'.
            target_images: Tensor of target images (N,3,H,W).
            patchsize: (height, width) in downsampled units.

        Returns:
            The modified UnbatchedExample after augmentation, cropping and pt3d prep.
        """
        # Augmentation (only for training when enabled in cfg).
        if self.stage == "train" and getattr(self.cfg, "augment", False):
            example = apply_augmentation_shim(example)

        # Cropping: compute full pixel shape from patchsize.
        height, width = patchsize
        example = apply_crop_shim(
            example=example,
            shape=(height * 14, width * 14),
            intr_aug=(self.stage == "train" and getattr(self.cfg, "intr_augment", False)),
        )

        # Prepare 3D points and normalize if requested by cfg.
        example = prepare_pts3d_and_normalize(
            example=example,
            target_images=target_images,
            normalize_by_pts3d=getattr(self.cfg, "normalize_by_pts3d", False),
        )

        return example
    def add_timestamps(
        self,
        example: UnbatchedExample,
        frames: list[SceneFrame],
        context_indices: Int64[Tensor, "views"],
        target_indices: Int64[Tensor, "views"],
    ) -> UnbatchedExample:
        """Add normalized timestamps to the example dict using 'num_frame' from frames metadata.
        Timestamps are normalized by the context's min-max."""

        if "num_frame" not in frames[0]:
            return example

        context_timestamps = torch.tensor(
            [frames[i].get("num_frame", 0) for i in context_indices], dtype=torch.float32
        )
        target_timestamps = torch.tensor(
            [frames[i].get("num_frame", 0) for i in target_indices], dtype=torch.float32
        )

        context_min_timestamp = context_timestamps.min()
        context_max_timestamp = context_timestamps.max()
        context_range = context_max_timestamp - context_min_timestamp
        if float(context_range) != 0.0:
            example["context"]["timestamp"] = (context_timestamps - context_min_timestamp) / context_range
            example["target"]["timestamp"] = (target_timestamps - context_min_timestamp) / context_range
        else:
            example["context"]["timestamp"] = torch.zeros_like(context_timestamps)
            example["target"]["timestamp"] = torch.zeros_like(target_timestamps)

        return example

    def expand_indices_for_cameras(
        self,
        time_indices: Int64[Tensor, "num_times"],
        num_cameras: int,
    ) -> Int64[Tensor, "num_times_x_cameras"]:
        """Expand time indices to all cameras."""
        return time_indices.repeat(num_cameras)

    def get_frames_for_sampler(self, scene: str) -> tuple[list[SceneFrame], torch.Tensor, torch.Tensor]:
        """Get frames to pass to view sampler.

        Looks up calibration data from the per-camera registry. For moving cameras
        (e.g., Ego4D), extrinsics are stored per-frame in SceneFrame; for static
        cameras, they are stored in the calibration registry.
        """
        frames = self.scenes_by_id[scene]
        calib = self.camera_calibrations[scene]
        extrinsics = torch.stack([
            calib[f["camera_id"]].get("extrinsics", f["extrinsics"]) for f in frames
        ])
        intrinsics = torch.stack([
            calib[f["camera_id"]]["intrinsics"] for f in frames
        ])
        return frames, extrinsics, intrinsics

    def get_num_cameras(self, scene: str) -> int:
        """Get the number of cameras in a scene. Override in multi-camera datasets."""
        return 1

    def getitem(
        self, index: int, num_context_views: int, patchsize: tuple[int, int],
        num_target_views: int = 0,
        num_cameras: int | None = None,
    ) -> UnbatchedExample:
        """Return an UnbatchedExample by sampling context and target views for a scene.

        Args:
            index (int): Dataset index that maps to a scene via self.index_to_scene.
            num_context_views (int): Number of context views to sample.
            patchsize (tuple[int, int]): Patch size in downsampled units
                (height, width) used for cropping and augmentation.
            num_target_views (int): Number of target views to sample (0 = use sampler default).

        Returns:
            UnbatchedExample: Dictionary with 'context' and 'target' entries.
        """

        scene = self.index_to_scene[index]
        frames = self.scenes_by_id[scene]

        # Propagate requested camera count to multi-camera datasets (e.g. Exo4D).
        if num_cameras is not None:
            self._requested_num_cameras = num_cameras

        # Get frames for sampler (single camera for multi-camera datasets)
        _, extrinsics, intrinsics = self.get_frames_for_sampler(scene)

        # Skip when field of view exceeds cfg.max_fov (if provided).
        max_fov = getattr(self.cfg, "max_fov", None)
        if max_fov is not None:
            fov = get_fov(intrinsics).rad2deg()
            if (fov > float(max_fov)).any():
                raise Exception(f"Field of view too wide: {fov}. Max: {max_fov}")

        try:
            context_indices, target_indices, _ = self.view_sampler.sample(
                scene=scene,
                num_context_views=num_context_views,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                num_target_views=num_target_views,
            )
        except ValueError:
            raise Exception("Not enough frames")

        context_frames, context_indices_expanded = self.get_frames_for_indices(
            scene=scene,
            time_indices=context_indices,
            is_context=True
        )
        target_frames, target_indices_expanded = self.get_frames_for_indices(
            scene=scene,
            time_indices=target_indices,
            is_context=False
        )

        # Load all frames in a single call and split by context count.
        n_context = len(context_frames)
        all_data = self.load_frames(context_frames + target_frames, scene)

        context_data: FrameData = {k: v[:n_context] for k, v in all_data.items()}  # type: ignore[assignment]
        target_data: FrameData = {k: v[n_context:] for k, v in all_data.items()}  # type: ignore[assignment]
        context_images = context_data["images"]
        context_extrinsics = context_data["extrinsics"]
        context_intrinsics = context_data["intrinsics"]
        target_images = target_data["images"]
        target_extrinsics = target_data["extrinsics"]
        target_intrinsics = target_data["intrinsics"]
        self.validate_image_shapes(context_images, target_images)

        # Combine context and target extrinsics for normalization
        combined_extrinsics = torch.cat([context_extrinsics, target_extrinsics], dim=0)
        combined_context_indices = torch.arange(len(context_extrinsics), dtype=torch.int64)
        combined_extrinsics, scale = self.normalize_extrinsics(
            extrinsics=combined_extrinsics,
            context_indices=combined_context_indices
        )

        # Split back into context and target
        context_extrinsics = combined_extrinsics[:len(context_frames)]
        target_extrinsics = combined_extrinsics[len(context_frames):]

        # Build scene name label
        cfg_name = getattr(self.cfg, "name", None)
        scene_label = f"{cfg_name}_{scene}" if cfg_name is not None else f"{scene}"
        num_cameras = getattr(self, "_num_cameras_used", 1)

        example: UnbatchedExample = {
            "context": {
                "extrinsics": context_extrinsics,
                "intrinsics": context_intrinsics,
                "image": context_images,
                "near": self.get_bound("near", len(context_extrinsics)) / scale,
                "far": self.get_bound("far", len(context_extrinsics)) / scale,
                "index": context_indices_expanded,
            },
            "target": {
                "extrinsics": target_extrinsics,
                "intrinsics": target_intrinsics,
                "image": target_images,
                "near": self.get_bound("near", len(target_extrinsics)) / scale,
                "far": self.get_bound("far", len(target_extrinsics)) / scale,
                "index": target_indices_expanded,
            },
            "scene": scene_label,
            "num_cameras": num_cameras
        }

        # Inject optional per-frame extras (tracks, visibility)
        for split, frame_data in (("context", context_data), ("target", target_data)):
            tracks = frame_data.get("tracks")
            visibility = frame_data.get("visibility")
            if tracks is not None and visibility is not None:
                example[split]["track"] = tracks
                example[split]["visibility"] = visibility

        # Add the timestamp informations if available
        example = self.add_timestamps(
            example=example,
            frames=frames,
            context_indices=context_indices,
            target_indices=target_indices,
        )

        # Apply augmentation / cropping / points normalization
        example = self.augment_example(example=example, target_images=target_images, patchsize=patchsize)
        V_ctx, C, H, W = example["context"]["image"].shape
        V_tgt = example["target"]["image"].shape[0]

        print(f"Sampled '{self.cfg.name}': {scene_label}' V_ctx: {V_ctx}, V_tgt: {V_tgt}, H: {H}, W: {W}.")

        return example

    def __getitem__(self, index_tuple: tuple[int, ...]) -> UnbatchedExample:
        """
        Generic __getitem__ that delegates to the dataset-specific getitem implementation.
        If an error occurs, retries with a random index.

        Args:
            index_tuple: Tuple (index, num_context_views, num_target_views, patchsize_h)
                or legacy (index, num_context_views, patchsize_h) or (index, num_context_views).

        Returns:
            UnbatchedExample: Dictionary with context and target data.
        """
        num_cameras = None
        if len(index_tuple) == 5:
            index, num_context_views, num_target_views, patchsize_h, num_cameras = index_tuple
        elif len(index_tuple) == 4:
            index, num_context_views, num_target_views, patchsize_h = index_tuple
        elif len(index_tuple) == 3:
            index, num_context_views, patchsize_h = index_tuple
            num_target_views = 0
        else:
            index, num_context_views = index_tuple
            patchsize_h = self.cfg.input_image_shape[0] // 14
            num_target_views = 0
        if num_target_views is None:
            num_target_views = 0
        patchsize_w = (self.cfg.input_image_shape[1] // 14)
        max_retries = int(getattr(self.cfg, "max_retry_per_item", 20))
        last_error: Exception | None = None
        for _ in range(max_retries):
            try:
                return self.getitem(index, num_context_views, (patchsize_h, patchsize_w), num_target_views, num_cameras)
            except Exception as e:
                last_error = e
                print(f"Error: {e}")
                index = np.random.randint(len(self))

        raise RuntimeError(
            f"Failed to sample a valid item after {max_retries} retries "
            f"(num_context_views={num_context_views}, num_target_views={num_target_views}). "
            f"Last error: {last_error}"
        )
        
    def __len__(self) -> int:
        return len(self.index_to_scene)


DatasetShim = Callable[[BaseDataset, Stage], BaseDataset]