#!/usr/bin/env python3
"""Preprocess EgoExo4D exocentric videos into per-sequence HDF5 files.

For each sequence: reads fisheye intrinsics/extrinsics from gopro_calibs.csv,
undistorts every frame, and writes a single .h5 file with frames, intrinsics,
extrinsics, and camera IDs.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from exo_utils import load_calibration, scale_intrinsics, write_hdf5


def compute_undistortion_maps(video_paths: dict[str, Path], calib: dict) -> dict[str, tuple]:
    """Pre-compute per-camera fisheye undistortion maps from the first frame of each video.

    Returns a dict mapping camera ID to a (map_x, map_y) tuple ready for cv2.remap.
    """
    undistortion_maps = {}
    for cam_id, video_path in video_paths.items():
        cap = cv2.VideoCapture(str(video_path))
        ok, first_frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read first frame for camera '{cam_id}'")

        frame_height, frame_width = first_frame.shape[:2]
        K    = calib[cam_id]["K"]
        dist = calib[cam_id]["dist"]
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            K, dist, np.eye(3), K,
            (frame_width, frame_height),
            cv2.CV_16SC2,
        )
        undistortion_maps[cam_id] = (map_x, map_y)

    return undistortion_maps


def prepare_cameras(data_dir: Path, sequence: str, calib: dict) -> dict[str, Path]:
    """Locate downscaled videos and rescale each camera's intrinsics to video resolution.

    Returns a dict mapping camera ID to its video file path.
    """
    video_dir = data_dir / "takes" / sequence / "frame_aligned_videos" / "downscaled" / "448"
    video_files = sorted(video_dir.glob("cam*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"no cam*.mp4 files found in {video_dir}")

    video_paths = {}
    for video_file in video_files:
        cam_id = video_file.stem
        if cam_id not in calib:
            raise KeyError(f"no calibration entry for camera '{cam_id}'")

        # Open briefly just to read the actual resolution, then close
        cap = cv2.VideoCapture(str(video_file))
        video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        calib[cam_id]["K"] = scale_intrinsics(calib[cam_id]["K"], video_width, video_height)
        video_paths[cam_id] = video_file

    return video_paths


def process_sequence(
    data_dir: Path,
    sequence: str,
    output_dir: Path,
    takes_data: list[dict] | None,
    silent: bool = False,
    idx: int = 0,
    total: int = 1,
) -> bool:
    """Process a single sequence: validate, load calibration, undistort, write HDF5.

    Returns True on success, False on any error.
    """
    tag = f"[{idx}/{total}] {sequence}"
    output_path = output_dir / sequence / f"{sequence}.h5"

    if output_path.exists():
        print(f"{tag}: already exists, skipping", flush=True)
        return True

    try:
        # Check the sequence is a known take in the dataset metadata
        if takes_data is None:
            takes_data = json.loads((data_dir / "takes.json").read_text())
        if not any(take.get("take_name") == sequence for take in takes_data):
            raise LookupError(f"'{sequence}' not found in takes.json")

        # Load per-camera intrinsics, distortion coefficients, and extrinsics
        calib = load_calibration(data_dir / "takes" / sequence / "trajectory")
        if calib is None:
            raise FileNotFoundError("gopro_calibs.csv not found")

        # Match video files to calibration entries and rescale intrinsics
        video_paths = prepare_cameras(data_dir, sequence, calib)

        # Compute undistortion maps before writing so write_hdf5 stays a pure I/O function
        undistortion_maps = compute_undistortion_maps(video_paths, calib)

        print(f"{tag}: processing {len(video_paths)} cameras", flush=True)
        num_frames = write_hdf5(video_paths, calib, undistortion_maps, output_path, silent=silent)
        print(f"{tag}: done — {num_frames} frames saved to {output_path}", flush=True)
        return True

    except Exception as error:
        print(f"{tag}: FAILED — {error}", flush=True)
        return False


def load_sequences(sequences_arg: str) -> list[str]:
    """Parse the --sequences argument into a list of sequence names.

    Accepts either a path to a text file (one name per line)
    or a comma-separated list of names passed directly on the command line.
    """
    path = Path(sequences_arg)
    if path.is_file():
        lines = path.read_text().splitlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return [name.strip() for name in sequences_arg.split(",") if name.strip()]


def run_parallel(args, sequences: list[str], takes_data: list[dict] | None) -> list[str]:
    """Run sequence processing across multiple worker processes. Returns failed sequence names."""
    failed = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all sequences upfront, then collect results as they complete
        futures = {
            executor.submit(
                process_sequence,
                args.data_dir, seq, args.output_dir, takes_data, True,
                idx + 1, len(sequences),
            ): seq
            for idx, seq in enumerate(sequences)
        }
        for future in as_completed(futures):
            seq = futures[future]
            try:
                if not future.result():
                    failed.append(seq)
            except Exception as error:
                print(f"EXCEPTION {seq}: {error}")
                failed.append(seq)
    return failed


def run_sequential(args, sequences: list[str], takes_data: list[dict] | None) -> list[str]:
    """Run sequence processing one at a time. Returns failed sequence names."""
    failed = []
    for idx, seq in enumerate(sequences):
        success = process_sequence(
            args.data_dir, seq, args.output_dir, takes_data, False,
            idx + 1, len(sequences),
        )
        if not success:
            failed.append(seq)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir",    type=Path, required=True, help="EgoExo4D dataset root (contains takes.json)")
    parser.add_argument("--output_dir",  type=Path, required=True, help="Output directory for HDF5 files")
    parser.add_argument("--sequences",   type=str,  required=True,
                        help="Path to a sequences file (one name per line) or a comma-separated list of names")
    parser.add_argument("--num_workers", type=int,  default=1,     help="Number of parallel worker processes (default: 1)")
    args = parser.parse_args()

    sequences = load_sequences(args.sequences)
    if not sequences:
        raise SystemExit("Error: no sequences provided")

    print(f"Data dir:  {args.data_dir}")
    print(f"Output:    {args.output_dir}")
    print(f"Sequences: {len(sequences)}")
    print(f"Workers:   {args.num_workers}\n")

    # Pre-load takes.json once and share it across all workers
    takes_path = args.data_dir / "takes.json"
    takes_data = json.loads(takes_path.read_text()) if takes_path.exists() else None

    if args.num_workers > 1:
        failed = run_parallel(args, sequences, takes_data)
    else:
        failed = run_sequential(args, sequences, takes_data)

    num_ok = len(sequences) - len(failed)
    print(f"\nDone: {num_ok}/{len(sequences)} sequences succeeded")
    if failed:
        print("Failed:\n" + "\n".join(f"  {seq}" for seq in failed))


if __name__ == "__main__":
    main()
