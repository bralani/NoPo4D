# Data: EgoExo4D Download & Preprocessing

These scripts download the EgoExo4D dataset and convert the raw videos into the HDF5 format expected by the NoPo4D dataloader.

## Download

### Setup

Install the AWS CLI and EgoExo4D downloader in one step:

```bash
pip install awscli ego4d --upgrade
```

Request your EgoExo4D access keys at https://ego4d.dev/request/ego-exo4d (can take up to 48 hours), then configure the CLI:

```bash
aws configure
```

`aws configure` prompts for your **Access Key ID** and **Secret Access Key** — press Enter twice to accept the default region and output format. See the [official AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html) for details. If you have multiple profiles, pass `--s3_profile <name>` to the `egoexo` command.

For the full EgoExo4D download documentation see https://docs.ego-exo4d-data.org/getting-started/.

### Run the download

```bash
export DATA_DIR=/your/dataset/path

egoexo -o $DATA_DIR \
  --parts metadata annotations captures take_trajectory ego_pose_pseudo_gt downscaled_takes/448 \
  --num_workers 12
```

Use `--universities georgiatech cmu unc sfu ...` to restrict the download to specific universities — the full dataset is very large. See https://docs.ego-exo4d-data.org/download/ for the full list of available parts and parameters.

## Preprocessing

`preprocess_exo.py` reads camera intrinsics/extrinsics, applies fisheye undistortion, and writes one HDF5 file per sequence containing all camera views.

**Warning: processing all sequences produces ~70 TB of data. Edit `scripts/dataset/sequences_train.txt` and `scripts/dataset/sequences_test.txt` to include only the sequences you need before running.**

### 0. Install preprocessing dependencies

The preprocessing script requires `opencv-python`, `h5py`, and `tqdm`, which are not part of the base NoPo4D install:

```bash
pip install opencv-python h5py tqdm
```

### 1. Define your splits

Fill in `scripts/dataset/sequences_train.txt` and `scripts/dataset/sequences_test.txt` with one sequence name per line (sequence names correspond to the `take_name` field in `takes.json`):

```
cmu_bike01_2
cmu_bike01_4
georgiatech_cooking_01_01_2
...
```

### 2. Run preprocessing

```bash
export DATA_DIR=/your/dataset/path
export OUTPUT_DIR=/your/dataset_processed/path

# Train split
bash scripts/dataset/preprocess_exo.sh scripts/dataset/sequences_train.txt <num_workers>

# Test split
bash scripts/dataset/preprocess_exo.sh scripts/dataset/sequences_test.txt <num_workers>
```

`num_workers` controls how many sequences are processed in parallel and defaults to 1. Set it to the number of available CPU cores to speed up preprocessing significantly.

### 3. Copy the split files into the processed dataset root

The dataloader looks for `sequences_train.txt` and `sequences_test.txt` directly inside the processed dataset root:

```bash
cp scripts/dataset/sequences_train.txt $OUTPUT_DIR/sequences_train.txt
cp scripts/dataset/sequences_test.txt  $OUTPUT_DIR/sequences_test.txt
```

### Output structure

```
<output_dir>/
├── sequences_train.txt
├── sequences_test.txt
└── <sequence_name>/
    └── <sequence_name>.h5
```

Each HDF5 file contains:

| Dataset | Shape | dtype | Description |
|---|---|---|---|
| `frames` | `[T, N, H, W, 3]` | float16 | Undistorted RGB frames |
| `intrinsics` | `[N, 3, 3]` | float64 | Camera intrinsic matrices |
| `extrinsics` | `[N, 4, 4]` | float64 | Camera-to-world poses |
| `camera_ids` | `[N]` | str | Camera identifiers (e.g. `cam01`) |

The dataloader selects the correct split file at runtime based on the `data_stage` field (`train` or `test`) passed through the experiment config.
