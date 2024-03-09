import cv2
import torch
import numpy as np

def build_targets(p, targets, anchors):
    """
    Args:
        p: tensor (batch_size, num_anchors, S, S, 5 + nc)
        targets: tensor (num_targets, 6) -> image_id, class_is, x, y, w, h
        anchors: tensor (3, 2) 

    """
    pred_shape = p.shape    # prediction shape
    na = anchors.shape[0]   # num of anchors
    nt =  targets.shape[0]  # num of targets

    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # to add anchors to targets
    targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)         # append anchort to targets: (na, n, 6) - > (na, n, 7)
    gain = torch.ones(7, device=targets.device)
    gain[2:6] = torch.tensor(pred_shape)[[3, 2, 3, 2]]                         # (1, 1, S, S, S, S, 1)   

    t = targets * gain   # normalize the target to grid size

    g = 0.5  # bias
    off = torch.tensor(
        [
            [0, 0],   # No offset (the center of the cell).
            [1, 0],   # Right by one cell
            [0, 1],   # Down by one cell
            [-1, 0],  # Left by one cell
            [0, -1],  # Up by one cell
        ],
        device=targets.device).float() * g  # offsets
    
    if nt:
        r = t[..., 4:6] / anchors[:, None]  # ratio of (width, height) / anchors
        
        """
        finds the maximum between r and 1 / r for each width and height. 
        This step is crucial because it ensures that both cases 
        (where the anchor is larger or smaller than the ground truth box) are considered equally
        """
        j = torch.max(r, 1 / r).max(2)[0]
        is_matched = j < anchor_t
        
        t = t[is_matched]

        # offset
        gxy = t[:, 2:4]  # extract x and y center coordinates of ground truth (normalized to grid size)
        gxy_inverse = gain[[2, 3]] - gxy   # Computes the inverse distances from the ground truth centers to the grid boundaries
        j, k = ((gxy % 1 < g) & (gxy > 1)).T   # check if the ground truth centers (gxy) are close to the grid boundaries and within a threshold g of the lower boundary of a grid cell
        l, m = ((gxy_inverse % 1 < g) & (gxy_inverse > 1)).T # same for upper boundriers
        j = torch.stack((torch.ones_like(j), j, k, l ,m))    # Creates a mask that combines the boundary conditions 
        t = t.repeat((5, 1, 1))[j] #  repeats the ground truth boxes for each of the boundary conditions (original, bottom, top, left, and right boundaries)
        offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # Calculates the offsets for the ground truth boxes that are close to the grid boundaries
    
    else:
        t = targets[0]
        offsets = 0 

    bc, gxy, gwh, a = t.chunk(4, 1)   # Splitting the Targets (t) into Components -> 1)  # (image, class), grid xy, grid wh, anchor
    a, (b, c) = a.long().view(-1), bc.long().T  
    gij = (gxy - offsets).long()
    gi, gj = gij.T

    return (
        c,
        torch.cat((gxy - gij, gwh), 1),
        anchors[a],
        (b, a, gj.clamp_(0, pred_shape[2] - 1), gi.clamp_(0, pred_shape[3] - 1))
    )

def draw_grids(image, grid_x, grid_y):
    """
    Draws grid lines on an image.

    Parameters:
    - image: The input image.
    - grid_x: Number of grids along the x-axis.
    - grid_y: Number of grids along the y-axis.

    Returns:
    - The image with the grid lines drawn.
    """
    h, w = image.shape[:2]  # Get the height and width of the image
    # Calculate the spacing for the grids
    spacing_x = w // grid_x
    spacing_y = h // grid_y
    
    # Draw the vertical lines
    for x in range(0, w, spacing_x):
        cv2.line(image, (x, 0), (x, h), color=(255, 255, 255), thickness=1)
    
    # Draw the horizontal lines
    for y in range(0, h, spacing_y):
        cv2.line(image, (0, y), (w, y), color=(255, 255, 255), thickness=1)

    return image

def draw_bounding_box_centers(image, boxes):

    """
    Draw the center of bounding boxes on an image. Assumes box coordinates are [xmin, ymin, xmax, ymax].

    Parameters:
    - image: The image array (numpy array).
    - boxes: A tensor or array of bounding boxes with shape [N, 4], where N is the number of boxes,
      and each box is defined as [xmin, ymin, xmax, ymax].

    Returns:
    - The image with drawn centers.

    """
    for box in boxes:
        # Calculate the center of the box
        center_x = int(((box[0] + box[2]) / 2) * image.shape[1] if box[0] < 1 else ((box[0] + box[2]) / 2))
        center_y = int((box[1] + box[3]) / 2 * image.shape[0] if box[1] < 1 else (box[1] + box[3]) / 2)

        # Draw the center on the image
        cv2.circle(image, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)

    return image

anchor_t = 4.0
anchors = torch.load('./assets/anchors.pt')
targets = torch.load('./assets/targets.pt')
imgs = torch.load('./assets/imgs.pt')
pred = torch.load('./assets/preds.pt')

print(targets.shape)
print(imgs.shape)

image_id = 7
image = imgs[image_id].permute(1, 2, 0).cpu().numpy()

image = cv2.resize(image, (1024, 1024))
target = targets[targets[..., 0] == image_id]



tcls, tbox, indices, anch = [], [], [], []
for i in range(len(pred)):
    c, box, an, indice = build_targets(pred[i], targets, anchors[i])
    break



pi = pred[i]
b, a, gj, gi = indice


print(b.shape)
print(a.shape)
print(gj.shape)
print(gi.shape)
print(pi.shape)
print(pi[b, a, gj, gi].shape)
pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, 2), 1)


bbx = []
for box in target:
    _, _, x, y, w, h = box
    x1, y1, x2, y2 = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)
    bbx.append((x1, y1, x2, y2))



    