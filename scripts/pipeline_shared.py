import os, sys
import numpy as np
import torch
import cv2

VGGT_REPO = r"C:\Users\great\cs6672\vggt"
sys.path.insert(0, VGGT_REPO)
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"  # cached in ~/.cache/torch/hub


def load(img_path, size=(518, 392), zoom=1.0):
    """
    Center crop to size's aspect, then keep the central 1/zoom of it. Principal point stays centered.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image from {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    target = size[0] / size[1]

    if w / h > target:
        cw, ch = h * target, h
    else:
        cw, ch = w, w / target

    cw, ch = round(cw / zoom), round(ch / zoom)
    x0, y0 = (w - cw) // 2, (h - ch) // 2

    out = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], size, interpolation=cv2.INTER_AREA)

    # Return the processed image as a PyTorch tensor per vggt.py 34
    return torch.from_numpy(out).float().permute(2, 0, 1) / 255.0


def build_model():
    model = VGGT(enable_track=False)
    weights = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu")
    info = model.load_state_dict(weights, strict=False)
    del weights
    assert not info.missing_keys, f"missing weights: {info.missing_keys[:5]}"

    # 8 GB card: backbone weights in bf16, heads stay fp32 (as in the original forward).
    # Convert on the CPU, before .cuda(), so the fp32 aggregator never sits on the GPU.
    agg = model.aggregator.to(torch.bfloat16)
    agg_forward = agg.forward

    # Heads run with autocast off, so hand them fp32 tokens; keep the None placeholders.
    agg.forward = lambda images: (
        lambda out: ([None if t is None else t.float() for t in out[0]], out[1])
    )(agg_forward(images.to(torch.bfloat16)))

    return model.cuda().eval()


@torch.no_grad()
def run(model, imgs):
    """
    imgs: S,3,H,W tensor in [0,1] (all one size).
    Returns per frame: R, t (world->camera, OpenCV), camera center C, K, fov (deg, [vertical, horizontal]),
    depth, dconf, pts (unprojected depth), pts_head (point head), pconf, imgs.
    """
    H, W = imgs.shape[-2:]

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(imgs.cuda())

    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
    E = extri[0].float().cpu().numpy()
    R, t = E[:, :, :3], E[:, :, 3]
    K = intri[0].float().cpu().numpy()
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()

    res = {
        "R": R,
        "t": t,
        "C": -np.einsum("sji,sj->si", R, t),  # C = -R^T t
        "K": K,
        "fov": np.degrees(pred["pose_enc"][0, :, 7:9].float().cpu().numpy()),
        "depth": depth,
        "dconf": pred["depth_conf"][0].float().cpu().numpy(),
        "pts": unproject_depth_map_to_point_map(depth[..., None], E, K).astype(np.float32),
        "pts_head": pred["world_points"][0].float().cpu().numpy(),
        "pconf": pred["world_points_conf"][0].float().cpu().numpy(),
        "imgs": imgs.cpu().numpy(),
    }

    del pred
    torch.cuda.empty_cache()
    return res


def ground_frame(res, idx):
    """
    Level frame from cameras idx. Rows of R are the camera axes in world coords (OpenCV: x right, y down, z forward).
    Returns origin, B (rows: right, forward, up) and scale, so (X - origin) @ B.T / scale is in camera-spread units.
    """
    R = res["R"][idx]
    up = -R[:, 1, :].mean(0)
    up /= np.linalg.norm(up)
    fwd = R[:, 2, :].mean(0)
    fwd -= fwd.dot(up) * up  # make it level
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)

    origin = res["C"][idx].mean(0)
    scale = np.median(np.linalg.norm(res["C"][idx] - origin, axis=1))
    return origin, np.stack([right, fwd, up]), scale


def to_ground(X, frame):
    """World points (..., 3) -> level frame, in units of camera spread."""
    origin, B, scale = frame
    return (X - origin) @ B.T / scale


def topdown(res, path, ref_idx=None, labels=None, colors=None, links=(), title="", size=900):
    """
    Top-down map: confident points of ref_idx frames in their own colors, every camera as a dot + view arrow.
    links: (a, b) pairs of frames to join with a line (e.g. a lens triple). Returns cameras in the level frame.
    """
    S = len(res["C"])
    ref_idx = list(range(S)) if ref_idx is None else list(ref_idx)
    labels = [str(i) for i in range(S)] if labels is None else labels
    colors = [(220, 40, 40)] * S if colors is None else colors  # RGB
    frame = ground_frame(res, ref_idx)

    conf = res["dconf"][ref_idx]
    m = conf > np.percentile(conf, 60)
    P = to_ground(res["pts"][ref_idx][m], frame)
    cols = (res["imgs"][ref_idx].transpose(0, 2, 3, 1)[m] * 255).astype(np.uint8)
    keep = np.random.default_rng(0).choice(len(P), min(80000, len(P)), replace=False)
    P, cols = P[keep], cols[keep]
    C = to_ground(res["C"], frame)
    F = res["R"][:, 2, :] @ frame[1].T  # view directions in the level frame

    # bounds: robust extent of the points, always including every camera
    lo = np.minimum(np.percentile(P[:, :2], 2, axis=0), C[:, :2].min(0)) - 0.5
    hi = np.maximum(np.percentile(P[:, :2], 98, axis=0), C[:, :2].max(0)) + 0.5
    sc = size / (hi - lo).max()
    top = 30  # room for the title
    to_px = lambda xy: (int((xy[0] - lo[0]) * sc), int(top + size - (xy[1] - lo[1]) * sc))  # flip y: rows grow down

    img = np.full((size + top, size, 3), 255, np.uint8)
    u = ((P[:, 0] - lo[0]) * sc).astype(int)
    v = (top + size - (P[:, 1] - lo[1]) * sc).astype(int)
    ok = (u >= 0) & (u < size) & (v >= top) & (v < size + top)
    img[v[ok], u[ok]] = cols[ok]

    for a, b in links:
        cv2.line(img, to_px(C[a]), to_px(C[b]), (0, 0, 0), 1, cv2.LINE_AA)
    for i in range(S):
        f = F[i, :2] / (np.linalg.norm(F[i, :2]) + 1e-9)
        cv2.arrowedLine(img, to_px(C[i]), to_px(C[i, :2] + 0.4 * f), colors[i], 2, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(img, to_px(C[i]), 5, colors[i], -1, cv2.LINE_AA)
        if labels[i]:
            x, y = to_px(C[i])
            cv2.putText(img, labels[i], (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[i], 1, cv2.LINE_AA)
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return C


def umeyama(src, dst):
    """
    Similarity transform (s, R, t) minimizing |dst - (s R src + t)|^2 over N,3 point sets.
    Use it to compare two runs: each run picks its own world frame (frame 0) and its own scale.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:  # otherwise we could get a reflection, not a rotation
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def apply_sim(sim, X):
    s, R, t = sim
    return s * X @ R.T + t


def overlap(res, i, others, radius_frac=0.01, n_query=5000, n_ref=80000):
    """
    Fraction of frame i's confident 3D points that land within radius_frac of the scene size
    of some confident point of the other frames. ~1.0 for a duplicate frame, ~0 for an unrelated image.
    """
    conf = res["dconf"]
    ref = res["pts"][others][conf[others] > np.percentile(conf[others], 50)]
    qry = res["pts"][i][conf[i] > np.percentile(conf[i], 50)]
    rng = np.random.default_rng(0)
    ref = ref[rng.choice(len(ref), min(n_ref, len(ref)), replace=False)]
    qry = qry[rng.choice(len(qry), min(n_query, len(qry)), replace=False)]

    extent = np.percentile(ref, 98, axis=0) - np.percentile(ref, 2, axis=0)
    r = radius_frac * np.linalg.norm(extent)

    A, B = torch.from_numpy(qry).cuda(), torch.from_numpy(ref).cuda()
    d = torch.cat([torch.cdist(A[k:k + 1000], B).min(1).values for k in range(0, len(A), 1000)])
    return float((d < r).float().mean())
