from typing import Callable, Literal
from typing_extensions import TypedDict, NotRequired, TypeAlias

from jaxtyping import Bool, Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]

# Named single-camera tensor types
Intrinsics: TypeAlias = Float[Tensor, "3 3"]
Extrinsics: TypeAlias = Float[Tensor, "4 4"]


# The following types mainly exist to make type-hinted keys show up in VS Code. Some
# dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call (e.g., the number of views,
#    which differs between context and target BatchedViews).


class BatchedViews(TypedDict):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    depth: NotRequired[Float[Tensor, "batch _ _ _"]]  # batch view height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view
    overlap: NotRequired[Float[Tensor, "batch _"]]  # batch view
    pts3d: NotRequired[Float[Tensor, "batch _ _ _ 3"]]  # batch view height width 3
    valid_mask: NotRequired[Bool[Tensor, "batch _ _ _"]]  # batch view height width
    timestamp: NotRequired[Float[Tensor, "batch _"]]  # batch view
    track: NotRequired[Float[Tensor, "batch _ _ 2"]]  # batch view num_tracks 2
    visibility: NotRequired[Bool[Tensor, "batch _ _"]]  # batch view num_tracks


class BatchedExample(TypedDict):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]
    num_cameras: list[int]


class UnbatchedViews(TypedDict):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ 3 height width"]
    depth: NotRequired[Float[Tensor, "_ height width"]]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]
    pts3d: NotRequired[Float[Tensor, "_ height width 3"]]
    valid_mask: NotRequired[Bool[Tensor, "_ height width"]]
    timestamp: NotRequired[Float[Tensor, " _"]]
    track: NotRequired[Float[Tensor, "_ num_tracks 2"]]
    visibility: NotRequired[Bool[Tensor, "_ num_tracks"]]


class UnbatchedExample(TypedDict):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str
    num_cameras: int


# Per-camera calibration stored once per camera (not duplicated per frame).
class CameraCalibration(TypedDict):
    intrinsics: Intrinsics
    extrinsics: NotRequired[Extrinsics]


# Per-frame metadata structure returned by scene index loaders.
class SceneFrame(TypedDict):
    file_path: NotRequired[str]      # PNG path
    h5_path: NotRequired[str]        # HDF5 file path
    num_frame: NotRequired[int]
    camera_id: str                   # Required: identifies which camera this frame is from
    extrinsics: NotRequired[Extrinsics]  # Per-frame only for moving cameras (Ego4D)


class FrameData(TypedDict):
    """Return type of BaseDataset.load_frames()."""
    images: Float[Tensor, "N 3 H W"]
    extrinsics: Float[Tensor, "N 4 4"]
    intrinsics: Float[Tensor, "N 3 3"]
    tracks: NotRequired[Float[Tensor, "N num_tracks 2"]]
    visibility: NotRequired[Bool[Tensor, "N num_tracks"]]


# A data shim modifies the example after it's been returned from the data loader.
DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews
