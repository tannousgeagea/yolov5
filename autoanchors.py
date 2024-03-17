import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from yolov5.dataset import Dataset
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans
PREFIX = 'Anchors: '

def kmeans_anchors(wh, k=9):
  """
  Performs K-means clustering on bounding boxes to find optimal anchor box sizes.

  Args:
      boxes: A numpy array of bounding boxes (num_boxes, 4) where each box is (x1, y1, x2, y2).
      k: The number of clusters (anchor boxes) to generate (default: 9).

  Returns:
      A numpy array of anchor boxes (k, 4) where each box is (w, h).
  """

  # kmeans = KMeans(n_clusters=k, random_state=0).fit(wh)
  # anchor_boxes = kmeans.cluster_centers_
  
  anchor_boxes, _ = kmeans(wh, k, iter=30) 
  return anchor_boxes

def iou_width_height(boxes1, boxes2):
    """
    Calculate the IoU of two sets of boxes that are in the center-size notation.
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    return intersection / union

def metric(wh, anchors, thr=4., iou=False):
  r = wh[:, None] / anchors
  x = torch.min(r, 1 / r).min(2)[0]
  if iou:
    x = iou_width_height(wh[:, None], anchors.view(-1, 2))
  
  best = x.max(1)[0]
  aat = (x > 1 / thr).float().sum(1).mean()
  bpr = (best > 1 / thr).float().mean()
  return aat, bpr

def kmeans_anchors(wh, k=9, iter=30, imgsz=640):
  try:
    anchors, _ = kmeans(wh, k, iter=iter)
    assert len(anchors) == k, f"not enough data to estimate an initial sets of anchor boxes"
  
  except:
    print('switchhing from kmeans to randomly assign anchor boxes')
    anchors = np.sort(np.random.rand(k * 2).reshape(k, 2)) * imgsz
  
  return anchors

def fitness_anchors(anchors, wh, thr=4.0, iou=False):
  r = wh[:, None] / anchors[None]
  x = torch.min(r, 1 / r).min(2)[0]
  if iou:
    x = iou_width_height(wh, anchors)
  best = x.max(1)[0]
  return (best * (best > 1 / thr).float()).mean()

def evolve(anchors, wh, f, gen=1000, mp=0.9, s=0.1, thr=4.0):
  pbar = tqdm(range(gen), ncols=125)

  sh = anchors.shape
  for _ in pbar:
    v = np.ones(sh)
    while (v == 1).all():
      v = ((np.random.random(sh) > mp) * random.random() * np.random.randn(*sh) * s + 1).clip(0.3, 3.0)

    new_anchors = (anchors.copy() * v).clip(min=2.0)
    fg = fitness_anchors(new_anchors, wh, thr=thr, iou=False)
    if fg > f:
      anchors, f = new_anchors.copy(), fg
      anchors = anchors[np.argsort(anchors.prod(1))]
      pbar.desc = f"{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}"

  return anchors

def autoanchors(dataset, thr=4.0, k=9, gen=1000, mp=0.9, s=0.1, iou=False):
  shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True) 
  wh0 = np.concatenate([l[:, 4:6] * s for l, s in zip(dataset.labels, shapes)])
  std = wh0.std(0)

  wh0 = torch.tensor(wh0).float()
  anchors = kmeans_anchors(wh0 / std, k, iter=30, imgsz=dataset.img_size) * std
  anchors = anchors[np.argsort(anchors.prod(1))]
  f =  fitness_anchors(anchors, wh, thr, iou)
  anchors = evolve(anchors, wh0, f, gen=gen, mp=mp, s=s, thr=thr)

  return anchors


imgsz = 640
dataset = Dataset(path='/home/appuser/data/animals.v2-release.yolov5pytorch', mode='train', imgsz=imgsz)
orig_anchors = torch.load('./yolov5/assets/anchors.pt')
anchors = orig_anchors
shapes = dataset.shapes
shapes = imgsz * shapes / shapes.max(1, keepdims=True)
scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
wh = torch.tensor(np.concatenate([l[:, 3:5] * s for l, s in zip(dataset.labels, shapes * scale)])).float()
stride = torch.tensor([8, 16, 32]).view(-1, 1, 1)
anchors = anchors * stride
aat, bpr = metric(wh, anchors.view(-1, 2), iou=False, thr=4.)
s = f"\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
if  bpr > 0.98:
  print(f"{s}Current anchors are a good fit to dataset ✅")
else:
  anchors = autoanchors(dataset)
  _, new_bpr = metric(wh, anchors)
  if new_bpr > bpr:  # replace anchors
      anchors = torch.tensor(anchors).type_as(orig_anchors)
      anchors = anchors.clone().view_as(orig_anchors)
      anchors /= stride
      s = f"{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)"
  else:
      s = f"{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)"
  print(s)
