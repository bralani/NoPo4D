import numpy as np
import torch
from evo.core.trajectory import PosePath3D

from src.utils.geometry import affine_inverse


def batch_align_poses_umeyama(ext_ref: torch.Tensor, ext_est: torch.Tensor):
    device, dtype = ext_ref.device, ext_ref.dtype
    assert ext_ref.dtype in [torch.float32, torch.float64]
    assert ext_est.dtype in [torch.float32, torch.float64]
    assert ext_ref.requires_grad is False
    assert ext_est.requires_grad is False
    rots, trans, scales = [], [], []
    for b in range(ext_ref.shape[0]):
        try:
            r, t, s = _align_poses_umeyama(ext_ref[b].cpu().numpy(), ext_est[b].cpu().numpy())
        except Exception:
            r = np.eye(3, dtype=np.float64)
            t = np.zeros(3, dtype=np.float64)
            s = 1.0
        rots.append(torch.from_numpy(r).to(device=device, dtype=dtype))
        trans.append(torch.from_numpy(t).to(device=device, dtype=dtype))
        scales.append(torch.tensor(s, device=device, dtype=dtype))
    return torch.stack(rots), torch.stack(trans), torch.stack(scales)


def apply_sim3_to_c2w(
    rots: torch.Tensor,
    trans: torch.Tensor,
    scales: torch.Tensor,
    poses_c2w: torch.Tensor,
) -> torch.Tensor:
    R_v = poses_c2w[..., :3, :3]
    t_v = poses_c2w[..., :3, 3]
    R = rots[:, None]
    s = scales[:, None, None]
    t = trans[:, None]
    new_R = R @ R_v
    new_t = s * (R @ t_v.unsqueeze(-1)).squeeze(-1) + t
    result = torch.eye(4, dtype=poses_c2w.dtype, device=poses_c2w.device)
    result = result.view(1, 1, 4, 4).expand(poses_c2w.shape[0], poses_c2w.shape[1], 4, 4).clone()
    result[..., :3, :3] = new_R
    result[..., :3, 3] = new_t
    return result


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Normalize all poses relative to a pivotal (reference) camera frame."""
    canonical = torch.eye(4, dtype=torch.float32, device=pivotal_pose.device).unsqueeze(0)
    norm_matrix = torch.bmm(canonical, torch.inverse(pivotal_pose))
    return torch.bmm(norm_matrix.expand(poses.shape[0], -1, -1), poses)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _affine_inverse_np(A: np.ndarray) -> np.ndarray:
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    axes = list(range(R.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    Rt = R.transpose(axes)
    return np.concatenate([np.concatenate([Rt, -Rt @ T], axis=-1), P], axis=-2)


def _to44(ext):
    if ext.shape[1] == 3:
        out = np.eye(4)[None].repeat(len(ext), 0)
        out[:, :3, :4] = ext
        return out
    return ext


def _poses_from_ext(ext_ref, ext_est):
    return _affine_inverse_np(_to44(ext_ref)), _affine_inverse_np(_to44(ext_est))


def _umeyama_sim3_from_paths(pose_ref, pose_est):
    path_ref = PosePath3D(poses_se3=pose_ref.copy())
    path_est = PosePath3D(poses_se3=pose_est.copy())
    r, t, s = path_est.align(path_ref, correct_scale=True)
    return r, t, s, np.stack(path_est.poses_se3)


def _apply_sim3_to_poses(poses, r, t, s):
    out = poses.copy()
    out[:, :3, :3] = r @ poses[:, :3, :3]
    out[:, :3, 3] = (r @ (s * poses[:, :3, 3].T)).T + t
    return out


def _median_nn_thresh(pose_ref, pose_est_aligned):
    P_ref = pose_ref[:, :3, 3]
    P_est = pose_est_aligned[:, :3, 3]
    dists = [np.linalg.norm(P_ref - p[None], axis=1).min() for p in P_est]
    return float(np.median(dists)) if dists else 0.0


def _align_poses_umeyama(ext_ref: np.ndarray, ext_est: np.ndarray, ransac=False, sub_n=None,
                          inlier_thresh=None, ransac_max_iters=10, random_state=None):
    pose_ref, pose_est = _poses_from_ext(ext_ref, ext_est)
    if not ransac:
        r, t, s, _ = _umeyama_sim3_from_paths(pose_ref, pose_est)
    else:
        r, t, s = _ransac_align_sim3(pose_ref, pose_est, sub_n=sub_n, inlier_thresh=inlier_thresh,
                                      max_iters=ransac_max_iters, random_state=random_state)
    return r, t, s


def _ransac_align_sim3(pose_ref, pose_est, sub_n=None, inlier_thresh=None, max_iters=10, random_state=None):
    rng = np.random.default_rng(random_state)
    N = pose_ref.shape[0]
    idx_all = np.arange(N)
    sub_n = max(3, min(sub_n, N)) if sub_n is not None else max(3, (N + 1) // 2)
    r0, t0, s0, pose_est0 = _umeyama_sim3_from_paths(pose_ref, pose_est)
    if inlier_thresh is None:
        inlier_thresh = _median_nn_thresh(pose_ref, pose_est0)
    P_ref_all = pose_ref[:, :3, 3]
    best_model = (r0, t0, s0)
    best_inliers = None
    best_score = (-1, np.inf)
    for _ in range(max_iters):
        sample = rng.choice(idx_all, size=sub_n, replace=False)
        try:
            r, t, s, _ = _umeyama_sim3_from_paths(pose_ref[sample], pose_est[sample])
        except Exception:
            continue
        errs = np.linalg.norm(_apply_sim3_to_poses(pose_est, r, t, s)[:, :3, 3] - P_ref_all, axis=1)
        inliers = errs <= inlier_thresh
        k = int(inliers.sum())
        mean_err = float(errs[inliers].mean()) if k > 0 else np.inf
        if (k > best_score[0]) or (k == best_score[0] and mean_err < best_score[1]):
            best_score = (k, mean_err)
            best_model = (r, t, s)
            best_inliers = inliers
    if best_inliers is not None and best_inliers.sum() >= 3:
        r, t, s, _ = _umeyama_sim3_from_paths(pose_ref[best_inliers], pose_est[best_inliers])
    else:
        r, t, s = best_model
    return r, t, s
