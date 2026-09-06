"""Dataset example types and TypedDict schemas."""

from typing import Literal
from typing_extensions import TypedDict

from jaxtyping import Float
from torch import Tensor

Stage = Literal["train", "val"]


class BatchedViews(TypedDict):
    extrinsics: Float[Tensor, "batch view 4 4"]
    intrinsics: Float[Tensor, "batch view 3 3"]
    image:      Float[Tensor, "batch view channel height width"]
    timestamp:  Float[Tensor, "batch view"] | None


class BatchedExample(TypedDict):
    target:      BatchedViews
    context:     BatchedViews
    scene:       list[str]
    num_cameras: list[int]


class UnbatchedViews(TypedDict):
    extrinsics: Float[Tensor, "view 4 4"]
    intrinsics: Float[Tensor, "view 3 3"]
    image:      Float[Tensor, "view 3 height width"]
    timestamp:  Float[Tensor, " view"] | None


class UnbatchedExample(TypedDict):
    target:      UnbatchedViews
    context:     UnbatchedViews
    scene:       str
    num_cameras: int


class CameraCalibration(TypedDict):
    intrinsics: Float[Tensor, "3 3"]
    extrinsics: Float[Tensor, "4 4"] | None  # None for datasets without pose data


class SceneFrame(TypedDict):
    scene_path: str | None                    # path to the source file (HDF5, PNG dir, etc.)
    num_frame:  int | None                    # temporal index within the scene
    camera_id:  str                           # which camera this frame belongs to
    extrinsics: Float[Tensor, "4 4"] | None   # per-frame pose, for datasets with moving cameras


SceneCameras = dict[str, CameraCalibration]  # {cam_id -> calibration} for all cameras in a scene
CameraFrames = dict[str, list[SceneFrame]]   # {cam_id -> frames}      for all cameras in a scene


class FrameData(TypedDict):
    """Stacked output of BaseDataset.load_frames(); N matches the input frame list length."""
    images:     Float[Tensor, "N 3 H W"]
    extrinsics: Float[Tensor, "N 4 4"]
    intrinsics: Float[Tensor, "N 3 3"]


AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews
