import os, sys
import numpy as np
import torch
import cv2

VGGT_REPO = r"C:\Users\great\cs6672\vggt"

def load(img_path,size = (518, 392), zoom=1.0):
    """
     Center crop to size's aspect, then keep the central 1/zoom of it. Principal point stays centered.
    """
    try:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    except:
        raise ValueError(f"Could not read image from {img_path}")

    h,w,_  = img.shape

    target = size[0]/size[1]

    if w / h > target:
        cw, ch = h * target, h
    else:
        cw, ch = w, w / target

    cw, ch = round(cw / zoom), round(ch / zoom)
    x0, y0 = (w - cw) // 2, (h - ch) // 2

    out = cv2.resize(img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)], size, interpolation=cv2.INTER_CUBIC)

    # Return the processed image as a PyTorch tensor per vggt.py 34
    return torch.from_numpy(out).float().permute(2, 0, 1) / 255.0

def build_model():
    sys.path.insert(0, VGGT_REPO)
    from vggt.models.vggt import VGGT

    model = VGGT(enable_track=False)
    