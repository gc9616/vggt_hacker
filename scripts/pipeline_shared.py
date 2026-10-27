import os, sys
import numpy as np
import torch
import cv2

REPO = r"C:\Users\great\cs6672\vggt"

def load(img_path, zoom=1.0):
    """
     Center crop to size's aspect, then keep the central 1/zoom of it. Principal point stays centered.
    """
    img = cv2.imread(img_path)
    h,w,_  = img.shape[:2]

    target = 518/392

    if w / h > target:
        cw, ch = h * target, h
    else:
        cw, ch = w, w / target

    cw, ch = cw / zoom, ch / zoom
    x0, y0 = (w - cw) / 2, (h - ch) / 2

    out = cv2.resize(img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)], (518, 392), interpolation=cv2.INTER_CUBIC)