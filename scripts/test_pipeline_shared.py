"""
Checkpoints for the shared pipeline (S1-S5) on the repo's kitchen scene.
Usage: python -u -W ignore test_pipeline_shared.py [n_frames]   (default 8)
"""
import os, sys, glob, time
import numpy as np
import torch

from pipeline_shared import (VGGT_REPO, load, build_model, run, ground_frame, topdown,
                             umeyama, apply_sim, overlap)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "pipeline_check")


def check_run(model, imgs):
    """S1-S2: one run, memory and sanity."""
    n = len(imgs)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = run(model, imgs)
    print(f"\n[S2] {n} frames in {time.time() - t0:.1f}s, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("  C[0] (should be ~0):", res["C"][0].round(4))
    print("  fov h per frame:", res["fov"][:, 1].round(1).tolist())
    print("  all finite:", all(np.isfinite(v).all() for v in res.values()))
    return res


def check_topdown(res):
    """S3: cameras should form an arc around the table, arrows pointing inward."""
    png = os.path.join(OUT, "kitchen_topdown.png")
    topdown(res, png, title=f"kitchen, {len(res['C'])} frames, units = camera spread")
    print(f"\n[S3] top-down map -> {os.path.abspath(png)}")


def check_umeyama(model, imgs, res):
    """S4: recover a known transform exactly, then align two real runs with different frame 0."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))  # a rotation, not a reflection
    s, R, t = umeyama(X, 2.5 * X @ q.T + [1, -2, 3])
    print(f"\n[S4] synthetic: scale {s:.6f} (2.5), rotation err {np.abs(R - q).max():.1e}, t {t.round(6)}")

    n = len(imgs)
    order = [3] + [i for i in range(n) if i != 3]  # same frames, frame 3 first
    res2 = run(model, imgs[order])
    sim = umeyama(res2["C"], res["C"][order])
    _, _, scale = ground_frame(res, list(range(n)))
    resid = np.linalg.norm(apply_sim(sim, res2["C"]) - res["C"][order], axis=1) / scale
    print(f"  run with frame 3 first, aligned to the first run: scale {sim[0]:.3f}, "
          f"camera residual median {np.median(resid):.3f}, max {resid.max():.3f} (camera-spread units)")


def check_overlap(model, imgs, res):
    """S5: a duplicate should score ~1; a held-out real frame gives the honest ceiling."""
    n = len(imgs)
    res_dup = run(model, torch.cat([imgs, imgs[1:2]]))
    dup = overlap(res_dup, n, list(range(n)))
    held = [overlap(res, i, [j for j in range(n) if j != i]) for i in range(1, n)]
    print(f"\n[S5] duplicate of frame 1: {dup:.3f}")
    print(f"  held-out real frames 1..{n - 1}: {np.round(held, 3).tolist()}, median {np.median(held):.3f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    paths = sorted(glob.glob(os.path.join(VGGT_REPO, "examples", "kitchen", "images", "*")))
    model = build_model()
    print("aggregator", next(model.aggregator.parameters()).dtype,
          "| heads", next(model.depth_head.parameters()).dtype,
          "| weights on GPU %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

    imgs = torch.stack([load(p) for p in paths[:n]])
    res = check_run(model, imgs)
    check_topdown(res)
    check_umeyama(model, imgs, res)
    check_overlap(model, imgs, res)
