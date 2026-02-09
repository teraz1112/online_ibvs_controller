from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from config import get_config
from src.correspondence import compute_correspondences, filter_correspondences_for_display
from src.cropper import crop_to_match
from src.dashboard import collect_flow_on_observed_records
from src.io_tools import load_image
from src.mask_to_picks import mask_to_picks

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class TwistPublisher:
    def __init__(self, enabled: bool, host: str, port: int, topic_name: str):
        self.enabled = enabled
        self._client = None
        self._topic = None
        if not enabled:
            return
        try:
            import roslibpy  # type: ignore
        except Exception as e:
            raise RuntimeError("roslibpy is required when --ros-enabled is set.") from e

        self._client = roslibpy.Ros(host=host, port=port)
        self._client.run()
        self._topic = roslibpy.Topic(self._client, topic_name, "geometry_msgs/Twist")
        self._roslibpy = roslibpy

    def publish(self, lx, ly, lz, ax, ay, az):
        if not self.enabled:
            return
        assert self._topic is not None
        self._topic.publish(
            self._roslibpy.Message(
                {
                    "linear": {"x": float(lx), "y": float(ly), "z": float(lz)},
                    "angular": {"x": float(ax), "y": float(ay), "z": float(az)},
                }
            )
        )

    def close(self):
        if self.enabled and self._client is not None:
            self._client.terminate()


@dataclass
class DummyGridFeatures:
    patch_feat: torch.Tensor
    gh: int
    gw: int
    shown_size: tuple[int, int]


class DummyExtractor:
    def extract(self, img: Image.Image, vit_patch: int, long_side: int, pad_to_multiple: bool):
        if long_side > 0:
            w, h = img.size
            scale = long_side / max(w, h)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img_r = img.resize(new_size, Image.BICUBIC)
        else:
            img_r = img.copy()

        w, h = img_r.size
        if pad_to_multiple:
            w_pad = ((w + vit_patch - 1) // vit_patch) * vit_patch
            h_pad = ((h + vit_patch - 1) // vit_patch) * vit_patch
            if (w_pad, h_pad) != (w, h):
                canvas = Image.new("RGB", (w_pad, h_pad))
                canvas.paste(img_r, (0, 0))
                img_x = canvas
            else:
                img_x = img_r
        else:
            img_x = img_r

        arr = np.asarray(img_x).astype(np.float32) / 255.0
        gh = arr.shape[0] // vit_patch
        gw = arr.shape[1] // vit_patch
        feats = []
        for qy in range(gh):
            for qx in range(gw):
                y0 = qy * vit_patch
                y1 = y0 + vit_patch
                x0 = qx * vit_patch
                x1 = x0 + vit_patch
                patch = arr[y0:y1, x0:x1, :]
                mean_rgb = patch.mean(axis=(0, 1))
                std_rgb = patch.std(axis=(0, 1))
                pos = np.array([(qx + 0.5) / gw, (qy + 0.5) / gh], dtype=np.float32)
                feats.append(np.concatenate([mean_rgb, std_rgb, pos], axis=0))
        feat = torch.tensor(np.asarray(feats), dtype=torch.float32)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return DummyGridFeatures(patch_feat=feat, gh=gh, gw=gw, shown_size=img_r.size)


def _grab_observed_from_camera(save_dir: Path | None = None) -> Image.Image:
    try:
        from pypylon import pylon  # type: ignore
    except Exception as e:
        raise RuntimeError("pypylon is required when --use-camera is set.") from e

    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("Basler camera not found.")

    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    camera.Open()
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    camera.StartGrabbingMax(1)
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    try:
        if not grab_result.GrabSucceeded():
            raise RuntimeError("Failed to grab frame from camera.")
        image = converter.Convert(grab_result)
        frame_bgr = image.Array
        frame_rgb = frame_bgr[..., ::-1].copy()
        pil_img = Image.fromarray(frame_rgb)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "captured_image.png"
            count = 1
            while save_path.exists():
                save_path = save_dir / f"captured_image_{count}.png"
                count += 1
            pil_img.save(save_path)
        return pil_img
    finally:
        grab_result.Release()
        camera.Close()


def write_video_from_frames(frames_dir: Path, out_mp4_path: Path, fps: int = 10):
    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        print("[WARN] No frames for video.")
        return
    if cv2 is None:
        print("[WARN] cv2 not available; skip mp4 writing.")
        return

    img0 = cv2.imread(str(paths[0]))
    if img0 is None:
        print("[WARN] Failed to read first frame; skip video.")
        return
    h, w = img0.shape[:2]
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4_path), fourcc, float(fps), (w, h))
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        vw.write(img)
    vw.release()


def _extract_point_pairs_for_epi(rows):
    if not rows:
        return None, None
    tgt_x_keys = ["src_x_tgt", "src_x_target", "src_x_tgt_disp"]
    tgt_y_keys = ["src_y_tgt", "src_y_target", "src_y_tgt_disp"]
    obs_x_keys = ["dst_x_obs", "dst_x", "dst_x_obs_disp"]
    obs_y_keys = ["dst_y_obs", "dst_y", "dst_y_obs_disp"]

    def pick_key(cands):
        for k in cands:
            if k in rows[0]:
                return k
        return None

    k_tx = pick_key(tgt_x_keys)
    k_ty = pick_key(tgt_y_keys)
    k_ox = pick_key(obs_x_keys)
    k_oy = pick_key(obs_y_keys)
    if None in (k_tx, k_ty, k_ox, k_oy):
        return None, None

    pts_t = np.array([[float(r[k_tx]), float(r[k_ty])] for r in rows], dtype=np.float64)
    pts_o = np.array([[float(r[k_ox]), float(r[k_oy])] for r in rows], dtype=np.float64)
    return pts_t, pts_o


def ransac_inlier_ratio_and_epipolar_median(rows, ransac_thr_px=1.0, confidence=0.999, max_iters=5000):
    pts_t, pts_o = _extract_point_pairs_for_epi(rows)
    if pts_t is None or pts_o is None:
        return np.nan, np.nan
    if len(pts_t) < 8 or cv2 is None:
        return np.nan, np.nan

    F, mask = cv2.findFundamentalMat(
        pts_t,
        pts_o,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=float(ransac_thr_px),
        confidence=float(confidence),
        maxIters=int(max_iters),
    )
    if F is None or mask is None:
        return np.nan, np.nan

    mask = mask.reshape(-1).astype(bool)
    inlier_ratio = float(mask.mean()) if len(mask) > 0 else np.nan
    if mask.sum() < 8:
        return inlier_ratio, np.nan

    x1 = np.hstack([pts_t, np.ones((len(pts_t), 1), dtype=np.float64)])
    x2 = np.hstack([pts_o, np.ones((len(pts_o), 1), dtype=np.float64)])
    l2 = (F @ x1.T).T
    l1 = (F.T @ x2.T).T
    denom2 = np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2) + 1e-12
    denom1 = np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2) + 1e-12
    d2 = np.abs(np.sum(l2 * x2, axis=1)) / denom2
    d1 = np.abs(np.sum(l1 * x1, axis=1)) / denom1
    d = 0.5 * (d1 + d2)
    d_epi = float(np.median(d[mask]))
    return inlier_ratio, d_epi


def mad_inlier_mask_from_rows(rows, k=2.0, min_keep=8):
    if not rows:
        return []
    r = np.array([np.hypot(float(rr["dx"]), float(rr["dy"])) for rr in rows], dtype=np.float64)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    z = (r - med) / mad
    mask = np.abs(z) <= float(k)
    if mask.sum() < min_keep and len(rows) >= min_keep:
        idx = np.argsort(np.abs(z))[:min_keep]
        mask = np.zeros_like(mask, dtype=bool)
        mask[idx] = True
    return mask.tolist()


def step_ratio_from_median_pixel_error(err_pix: float, s_min=0.0035, s_max=1.0, e0=250.0, p=2.0) -> float:
    x = float(err_pix) / float(e0)
    s = s_min + (s_max - s_min) * np.exp(-(x**float(p)))
    return float(np.clip(s, s_min, s_max))


def damping_from_error(err_pix: float, mu_min=1.0e-4, mu_max=5.0e-2, e0=250.0, p=2.0) -> float:
    x = float(err_pix) / float(e0)
    a = 1.0 - np.exp(-(x**float(p)))
    mu = mu_min + (mu_max - mu_min) * a
    return float(np.clip(mu, mu_min, mu_max))


def scale_intrinsics_to_size(fx, fy, cx, cy, calib_w, calib_h, w, h):
    sx = float(w) / float(calib_w)
    sy = float(h) / float(calib_h)
    return fx * sx, fy * sy, cx * sx, cy * sy


def keep_topn_rows_by_score(rows, topn: int | None):
    if (topn is None) or (topn <= 0) or (rows is None) or (len(rows) <= topn):
        return rows
    score_keys = ["score", "sim", "cos", "match_score", "confidence"]

    def pick_key():
        if not rows:
            return None
        for k in score_keys:
            if k in rows[0]:
                return k
        return None

    k = pick_key()
    if k is None:
        return rows

    def safe_score(r):
        try:
            return float(r.get(k, -1e18))
        except Exception:
            return -1e18

    rows_sorted = sorted(rows, key=safe_score, reverse=True)
    return rows_sorted[:topn]


def fit_pca_obs_from_target_mask(Ft: torch.Tensor, idx_list: list[int], n: int):
    X = Ft[idx_list].to(torch.float32)
    K, D = X.shape
    if K < 2:
        raise RuntimeError(f"PCA fit needs >=2 samples, got K={K}")
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    q_max = min(K - 1, D)
    n_eff = min(int(n), q_max)
    q = max(n_eff, min(32, q_max))
    _, _, V = torch.pca_lowrank(Xc, q=q, center=False)
    W = V[:, :n_eff].contiguous()
    return W, mu.squeeze(0), n_eff


def project_to_pca_obs(F: torch.Tensor, W: torch.Tensor, mu: torch.Tensor):
    Z = (F.to(torch.float32) - mu) @ W
    return torch.nn.functional.normalize(Z, dim=-1)


@dataclass
class StaticTargetContext:
    cfg: Any
    target_path: Path
    mask_path: Path
    use_dummy_extractor: bool = False

    def __post_init__(self):
        if not self.target_path.exists():
            raise FileNotFoundError(f"Target image not found: {self.target_path}")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask image not found: {self.mask_path}")

        self.img_tgt = load_image(self.target_path)
        self.img_mask = load_image(self.mask_path)
        if self.use_dummy_extractor:
            self.extractor = DummyExtractor()
        else:
            from src.encoders import get_feature_extractor

            self.extractor = get_feature_extractor(
                self.cfg.encoder,
                self.cfg.model_name,
                feat_mode=self.cfg.feat_mode,
                feat_layer=self.cfg.feat_layer,
                feat_last_k=self.cfg.feat_last_k,
            )
            if self.cfg.encoder.lower() == "ours":
                self.extractor.shallow_layer = int(self.cfg.ours_shallow_layer)
                self.extractor.clip_feat_mode = self.cfg.ours_clip_feat_mode
                self.extractor.clip_feat_layer = int(self.cfg.ours_clip_feat_layer)
                self.extractor.clip_feat_last_k = int(self.cfg.ours_clip_feat_last_k)
                self.extractor.use_dino_deep = bool(self.cfg.ours_use_dino_deep)
                self.extractor.use_dino_shallow = bool(self.cfg.ours_use_dino_shallow)
                self.extractor.use_clip = bool(self.cfg.ours_use_clip)

        self.gf_t = self.extractor.extract(
            img=self.img_tgt,
            vit_patch=self.cfg.vit_patch,
            long_side=self.cfg.long_side,
            pad_to_multiple=self.cfg.pad_to_multiple,
        )
        picks = mask_to_picks(mask_img=self.img_mask, gf_t=self.gf_t, min_covered_ratio=0.0)
        if len(picks) == 0:
            qy_mid, qx_mid = self.gf_t.gh // 2, self.gf_t.gw // 2
            picks = [(qy_mid, qx_mid, qy_mid * self.gf_t.gw + qx_mid)]
        self.picks = picks
        self.idx_list = [idx for (_, _, idx) in self.picks]

        self.pca_W = None
        self.pca_mu = None
        if self.cfg.encoder.lower() == "ours" and self.cfg.ours_pca_dim is not None:
            Ft_raw = self.gf_t.patch_feat
            W, mu, n_eff = fit_pca_obs_from_target_mask(Ft_raw, self.idx_list, n=int(self.cfg.ours_pca_dim))
            self.gf_t.patch_feat = project_to_pca_obs(Ft_raw, W, mu)
            self.pca_W, self.pca_mu = W, mu
            print(f"[INFO] PCA_obs: K={len(self.idx_list)}, raw_dim={Ft_raw.shape[1]} -> n={n_eff}")

        Wt, Ht = self.gf_t.shown_size
        self.tgt_disp = self.img_tgt if self.img_tgt.size == (Wt, Ht) else self.img_tgt.resize((Wt, Ht))


def process_one_observation(ctx: StaticTargetContext, observed_path: Path | None, use_camera: bool, camera_dump_dir: Path | None):
    cfg = ctx.cfg
    if use_camera:
        img_obs = _grab_observed_from_camera(save_dir=camera_dump_dir)
    else:
        if observed_path is None or not observed_path.exists():
            raise FileNotFoundError(f"Observed image not found: {observed_path}")
        img_obs = load_image(observed_path)

    cropped, _ = crop_to_match(target_img=ctx.img_tgt, observed_img=img_obs, mode=cfg.crop_mode)
    gf_c = ctx.extractor.extract(
        img=cropped,
        vit_patch=cfg.vit_patch,
        long_side=cfg.long_side,
        pad_to_multiple=cfg.pad_to_multiple,
    )
    if ctx.pca_W is not None:
        gf_c.patch_feat = project_to_pca_obs(gf_c.patch_feat, ctx.pca_W, ctx.pca_mu)

    Wc, Hc = gf_c.shown_size
    crp_disp = cropped if cropped.size == (Wc, Hc) else cropped.resize((Wc, Hc))
    flow_arrows_full = compute_correspondences(
        gf_t=ctx.gf_t,
        gf_c=gf_c,
        picks=ctx.picks,
        mode=cfg.match_mode,
        min_sim=cfg.mutual_min_sim,
        topk=cfg.mutual_topk_K,
    )
    flow_arrows_display = filter_correspondences_for_display(
        flow_arrows_full,
        stride=cfg.flow_stride,
        min_score=cfg.flow_min_score,
    )
    rows = collect_flow_on_observed_records(target_img=ctx.tgt_disp, cropped_img=crp_disp, flow_arrows=flow_arrows_display)
    return crp_disp, rows


def pixels_to_normalized(pts_uv, fx, fy, cx, cy):
    pts_uv = np.asarray(pts_uv, dtype=np.float64)
    u = pts_uv[:, 0]
    v = pts_uv[:, 1]
    x = (u - cx) / fx
    y = (v - cy) / fy
    return np.stack([x, y], axis=1)


def build_interaction_matrix(pts_xy, depths):
    pts_xy = np.asarray(pts_xy, dtype=np.float64)
    N = pts_xy.shape[0]
    depths = np.asarray(depths, dtype=np.float64)
    if depths.ndim == 0:
        depths = np.full(N, float(depths), dtype=np.float64)
    L_blocks = []
    for i in range(N):
        x, y = pts_xy[i]
        Z = depths[i]
        L_i = np.array(
            [
                [-1.0 / Z, 0.0, x / Z, x * y, -(1.0 + x**2), y],
                [0.0, -1.0 / Z, y / Z, 1.0 + y**2, -x * y, -x],
            ],
            dtype=np.float64,
        )
        L_blocks.append(L_i)
    return np.vstack(L_blocks)


def solve_wdls(L, b, mu, rot_reg_xyz):
    L = np.asarray(L, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    AtA = L.T @ L
    rhs = L.T @ b
    I = np.eye(AtA.shape[0], dtype=np.float64)
    R = np.zeros_like(AtA)
    rot_x, rot_y, rot_z = rot_reg_xyz
    R[3, 3] = float(rot_x)
    R[4, 4] = float(rot_y)
    R[5, 5] = float(rot_z)
    return np.linalg.solve(AtA + (mu * mu) * I + R, rhs)


def estimate_camera_twist_micro_step(
    pts_obs_uv,
    pts_tgt_uv,
    fx,
    fy,
    cx,
    cy,
    depths,
    step_ratio,
    mu,
    rot_reg_xyz,
    dt=1.0,
):
    pts_obs_uv = np.asarray(pts_obs_uv, dtype=np.float64)
    pts_tgt_uv = np.asarray(pts_tgt_uv, dtype=np.float64)
    pts_obs_xy = pixels_to_normalized(pts_obs_uv, fx, fy, cx, cy)
    pts_tgt_xy = pixels_to_normalized(pts_tgt_uv, fx, fy, cx, cy)
    s = pts_obs_xy.flatten()
    s_star = pts_tgt_xy.flatten()
    d = s_star - s
    s_dot_des = (float(step_ratio) * d) / float(dt)
    L_s = build_interaction_matrix(pts_obs_xy, depths)
    return solve_wdls(L_s, s_dot_des, mu=float(mu), rot_reg_xyz=rot_reg_xyz)


def apply_deadband_and_saturation(cmd6, eps, v_max, w_max):
    cmd6 = np.asarray(cmd6, dtype=np.float64).copy()
    cmd6[np.abs(cmd6) < eps] = 0.0
    cmd6[0:3] = np.clip(cmd6[0:3], -v_max, v_max)
    cmd6[3:6] = np.clip(cmd6[3:6], -w_max, w_max)
    return cmd6


def _build_parser():
    parser = argparse.ArgumentParser(description="Online IBVS controller.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML runtime config.")
    parser.add_argument("--scenario", type=str, default="dino")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--observed", type=Path, default=None, help="Observed image used when --use-camera is not set.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--use-camera", action="store_true")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--topn-match", type=int, default=None)
    parser.add_argument("--fix-depth", type=float, default=0.2)
    parser.add_argument("--calib-w", type=int, default=1500)
    parser.add_argument("--calib-h", type=int, default=1500)
    parser.add_argument("--fx-calib", type=float, default=1500.0)
    parser.add_argument("--fy-calib", type=float, default=1500.0)
    parser.add_argument("--cx-calib", type=float, default=750.0)
    parser.add_argument("--cy-calib", type=float, default=750.0)
    parser.add_argument("--converge-epx", type=float, default=40.0)
    parser.add_argument("--converge-frames", type=int, default=5)
    parser.add_argument("--deadband-eps", type=float, default=1.0e-6)
    parser.add_argument("--v-max", type=float, nargs=3, default=[0.02, 0.02, 0.02])
    parser.add_argument("--w-max", type=float, nargs=3, default=[0.10, 0.10, 0.10])
    parser.add_argument("--ros-enabled", action="store_true")
    parser.add_argument("--ros-host", type=str, default="127.0.0.1")
    parser.add_argument("--ros-port", type=int, default=9090)
    parser.add_argument("--ros-topic", type=str, default="/torobo/delta_cmd")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--use-dummy-extractor", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    cfg_file = _load_yaml(args.config)
    runtime_cfg = cfg_file.get("run", {})

    scenario = runtime_cfg.get("scenario", args.scenario)
    cfg = get_config(scenario)

    target = Path(runtime_cfg.get("target", str(args.target)))
    mask = Path(runtime_cfg.get("mask", str(args.mask)))
    observed = runtime_cfg.get("observed", str(args.observed) if args.observed else None)
    observed_path = Path(observed) if observed else None

    output_dir = Path(runtime_cfg.get("output_dir", str(args.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = output_dir / "observed_frames"
    obs_dir.mkdir(parents=True, exist_ok=True)
    camera_dump_dir = output_dir / "camera_captures"
    camera_dump_dir.mkdir(parents=True, exist_ok=True)

    use_camera = bool(runtime_cfg.get("use_camera", args.use_camera))
    max_iter = int(runtime_cfg.get("max_iter", args.max_iter))
    topn_match = runtime_cfg.get("topn_match", args.topn_match)
    fix_depth = float(runtime_cfg.get("fix_depth", args.fix_depth))

    calib_w = int(runtime_cfg.get("calib_w", args.calib_w))
    calib_h = int(runtime_cfg.get("calib_h", args.calib_h))
    fx_calib = float(runtime_cfg.get("fx_calib", args.fx_calib))
    fy_calib = float(runtime_cfg.get("fy_calib", args.fy_calib))
    cx_calib = float(runtime_cfg.get("cx_calib", args.cx_calib))
    cy_calib = float(runtime_cfg.get("cy_calib", args.cy_calib))

    converge_epx = float(runtime_cfg.get("converge_epx", args.converge_epx))
    converge_frames = int(runtime_cfg.get("converge_frames", args.converge_frames))
    deadband_eps = float(runtime_cfg.get("deadband_eps", args.deadband_eps))
    v_max = np.array(runtime_cfg.get("v_max", args.v_max), dtype=np.float64)
    w_max = np.array(runtime_cfg.get("w_max", args.w_max), dtype=np.float64)

    ros_enabled = bool(runtime_cfg.get("ros_enabled", args.ros_enabled))
    ros_host = str(runtime_cfg.get("ros_host", args.ros_host))
    ros_port = int(runtime_cfg.get("ros_port", args.ros_port))
    ros_topic = str(runtime_cfg.get("ros_topic", args.ros_topic))

    fps = int(runtime_cfg.get("fps", args.fps))
    write_video = bool(runtime_cfg.get("write_video", args.write_video))
    use_dummy_extractor = bool(runtime_cfg.get("use_dummy_extractor", args.use_dummy_extractor))

    publisher = TwistPublisher(enabled=ros_enabled, host=ros_host, port=ros_port, topic_name=ros_topic)
    ctx = StaticTargetContext(cfg=cfg, target_path=target, mask_path=mask, use_dummy_extractor=use_dummy_extractor)

    target_out_path = output_dir / "target.png"
    init_obs_out_path = output_dir / "init_observed.png"
    final_obs_out_path = output_dir / "final_observed.png"
    video_out_path = output_dir / "observed.mp4"
    csv_out_path = output_dir / "valdata.csv"
    ctx.img_tgt.save(target_out_path)

    records = []
    frame_idx = 0
    consec_ok = 0
    last_crp_disp = None

    try:
        for _ in range(max_iter):
            t0 = time.perf_counter()

            crp_disp, rows = process_one_observation(
                ctx=ctx,
                observed_path=observed_path,
                use_camera=use_camera,
                camera_dump_dir=camera_dump_dir,
            )
            rows = keep_topn_rows_by_score(rows, topn_match)
            mask_row = mad_inlier_mask_from_rows(rows, k=float(cfg.mad_k), min_keep=8)
            rows = [r for r, m in zip(rows, mask_row) if m]
            if len(rows) == 0:
                print("[WARN] No valid feature rows. Skip iteration.")
                continue

            frame_idx += 1
            last_crp_disp = crp_disp
            frame_path = obs_dir / f"{frame_idx:06d}.png"
            crp_disp.save(frame_path)
            if frame_idx == 1:
                crp_disp.save(init_obs_out_path)

            target_x = np.array([r["src_x_obs_mapped"] for r in rows], dtype=np.float32)
            target_y = np.array([r["src_y_obs_mapped"] for r in rows], dtype=np.float32)
            obs_x = np.array([r["dst_x_obs"] for r in rows], dtype=np.float32)
            obs_y = np.array([r["dst_y_obs"] for r in rows], dtype=np.float32)

            dx_pix = obs_x - target_x
            dy_pix = obs_y - target_y
            err_pix = float(np.median(np.sqrt(dx_pix * dx_pix + dy_pix * dy_pix)))
            step_ratio_cur = step_ratio_from_median_pixel_error(err_pix)
            mu_cur = damping_from_error(err_pix, mu_min=1e-4, mu_max=5e-2, e0=250.0, p=2.0)

            rot_min = 1e-6
            rot_max = 1e-2
            a = np.clip(mu_cur / 5e-2, 0.0, 1.0)
            rot_x = rot_min + (rot_max - rot_min) * (a**2)
            rot_reg_xyz = (rot_x, 0.0, 0.0)

            obs_w, obs_h = crp_disp.size
            fx_eff, fy_eff, cx_eff, cy_eff = scale_intrinsics_to_size(
                fx_calib, fy_calib, cx_calib, cy_calib, calib_w, calib_h, obs_w, obs_h
            )
            v_c = estimate_camera_twist_micro_step(
                pts_obs_uv=np.stack([obs_x, obs_y], axis=1),
                pts_tgt_uv=np.stack([target_x, target_y], axis=1),
                fx=fx_eff,
                fy=fy_eff,
                cx=cx_eff,
                cy=cy_eff,
                depths=fix_depth,
                step_ratio=step_ratio_cur,
                mu=mu_cur,
                rot_reg_xyz=rot_reg_xyz,
                dt=1.0,
            )
            e_norm = float(np.linalg.norm(v_c))
            e_trans = float(np.linalg.norm(v_c[0:3]))
            e_rot = float(np.linalg.norm(v_c[3:6]))
            N_match = int(len(rows))
            row_inlier, d_epi = ransac_inlier_ratio_and_epipolar_median(rows)
            conv_time = float(time.perf_counter() - t0)

            v_x = v_c[1]
            v_y = -v_c[0]
            v_z = v_c[2]
            w_x = v_c[4]
            w_y = -v_c[3]
            w_z = v_c[5]
            cmd = apply_deadband_and_saturation([v_x, v_y, v_z, w_x, w_y, w_z], eps=deadband_eps, v_max=v_max, w_max=w_max)
            publisher.publish(lx=cmd[0], ly=cmd[1], lz=cmd[2], ax=cmd[3], ay=cmd[4], az=cmd[5])

            records.append(
                {
                    "iter": frame_idx,
                    "est_vx": float(v_c[0]),
                    "est_vy": float(v_c[1]),
                    "est_vz": float(v_c[2]),
                    "est_wx": float(v_c[3]),
                    "est_wy": float(v_c[4]),
                    "est_wz": float(v_c[5]),
                    "conv_time": conv_time,
                    "e": err_pix,
                    "e_norm": e_norm,
                    "e_trans": e_trans,
                    "e_rot": e_rot,
                    "N_match": N_match,
                    "row_inlier": row_inlier,
                    "d_epi": d_epi,
                }
            )

            print(
                f"[INFO] iter={frame_idx}, err_pix={err_pix:.2f}px, "
                f"e_trans={e_trans:.6e}, e_rot={e_rot:.6e}, N_match={N_match}, inlier={row_inlier}"
            )
            ok = err_pix <= converge_epx
            consec_ok = (consec_ok + 1) if ok else 0
            if consec_ok >= converge_frames:
                print(f"[INFO] Converged at iter={frame_idx}")
                break

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received.")

    if last_crp_disp is not None:
        last_crp_disp.save(final_obs_out_path)
    df = pd.DataFrame(records)
    df.to_csv(csv_out_path, index=False)

    if write_video:
        write_video_from_frames(obs_dir, video_out_path, fps=fps)

    try:
        publisher.publish(0, 0, 0, 0, 0, 0)
    finally:
        publisher.close()
    print(f"[OK] Saved: {csv_out_path}")


if __name__ == "__main__":
    main()
