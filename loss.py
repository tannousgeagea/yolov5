import torch
import torch.nn as nn
from metrics import box_iou


class Loss(nn.Module):
    sort_obj_iou = False
    def __init__(self, anchors):
        super(Loss, self).__init__()
        self.anchors = anchors
        self.na = len(anchors)
        self.anchor_t = 4.0
        self.nc = 2
        self.gr = 1.0
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.cn, self.cp = 0.05, 0.95

    def forward(self, pred, targets, anchors):
        """
        forward method for yolov5 loss

        Args:
            - pred: (tensor) output of the model. (batch_size, anchors, grid_size, grid_size, 5 + num_of_class)
            - targets: (tensor) ground truth bounding boxed. (num of targets, 6) image_id, class_id, x, y, w, h

        Return:
            - loss: (tensor)
        """
        lcls = torch.zeros(1, device=pred.device)  # class loss
        lbox = torch.zeros(1, device=pred.device)  # box loss
        lobj = torch.zeros(1, device=pred.device)  # object loss

        bs = pred.shape[0]   # batch_size
        tobj = torch.zeros(pred.shape[:4], dtype=pred.dtype, device=pred.device)
        indices, tcls, tbox, anch = self.build_targets(pred, targets, anchors)
        b, a, gj, gi = indices
        pxy, pwh, _, pcls = pred[b, a, gj, gi].split((2, 2, 1, self.nc), dim=1)

        ### localization loss ###
        pxy = pxy.sigmoid() * 2 - 0.5
        pwh = (2 * pwh.sigmoid()) ** 2 * anch
        pbox = torch.cat((pxy, pwh), 1)
        iou = box_iou(pbox, tbox, xywh=True, CIoU=True)
        lbox += (1.0 - iou).mean()

        ### objectness loss ###
        """
        Detaches the IoU tensor from the current computation graph, 
        ensuring that the IoU scores are not involved in gradient computation. 
        This is necessary because we're using IoU scores for adjusting loss, not for backpropagation
        """
        iou = iou.detach().clamp(0).type(tobj.dtype)
        if self.sort_obj_iou:
            j = iou.argsort()
            b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
        if self.gr < 1:  # adjusts the IoU scores based on the gradient reduction 
            iou = (1.0 - self.gr) + iou * self.gr
        tobj[b, a, gj, gi] = iou.squeeze()
        lobj += self.bce(pred[..., 4], tobj)

        ### Classification ###
        n = b.shape[0]
        t = torch.full_like(pcls, self.cn, device=pcls.device)
        t[range(n), tcls] = self.cp
        lcls += self.bce(pcls, t)

        return (lbox + lcls + lobj) * bs, torch.cat((lbox, lcls, lobj)).detach() 
 
        
    def build_targets(self, pred, targets, anchors):
        """
        The build_targets method in YOLOv5 is essential for aligning the ground truth data with the model's 
        predictions during training. It assigns each ground truth bounding box to the appropriate grid cell 
        and anchor box across different scales (P3, P4, P5) of the model

        Args:
            - pred: (tensor) output of the model. (batch_size, anchors, grid_size, grid_size, 5 + num_of_class)
            - targets: (tensor) ground truth bounding boxed. (num of targets, 6) image_id, class_id, x, y, w, h

        Return:
            - set of matrices representing the targets for objectness, class, and bounding boxes at each scale
        """

        pred_shape = pred.shape
        nt = targets.shape[0]
        na = len(anchors)


        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt) # prepare to add anchors to targets | ai: [na, nt]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), -1) # add anchor index to target: (nt, 6) -> (na, nt, 7)         
        gain[2:6] = torch.tensor(pred_shape)[[3, 2, 3, 2]]    # scale only the index for x, y w, h
        

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

        t = targets * gain  # scale the target to grid size
        if nt:
            gwh = t[..., 4:6]   # extract width and heigh
            r = gwh / anchors[:, None]  # ratio of (width, height) / anchors

            """
            finds the maximum between r and 1 / r for each width and height. 
            This step is crucial because it ensures that both cases 
            (where the anchor is larger or smaller than the ground truth box) are considered equally
            """
            j = torch.max(r, 1 / r).max(2)[0] < self.anchor_t
            t = t[j]  # filter targets 

            
            """
            determining the assignment of ground truth bounding boxes to grid cells, 
            considering offsets to handle boundary cases.
            """
            gxy = t[..., 2:4]  # extract x and y center coordinates of ground truth (normalized to grid size)
            gxy_inverse = gain[[2, 3]] - gxy  # Computes the inverse distances from the ground truth centers to the grid boundaries
            j, k = ((gxy % 1 < g) & (gxy > 1)).T   # check if the ground truth centers (gxy) are close to the grid boundaries and within a threshold g of the lower boundary of a grid cell
            l, m = ((gxy_inverse % 1 < g) & (gxy_inverse > 1)).T  # same but for upper boundry
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat(5,  1, 1)[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

        else:
            t = targets[0]
            offsets = 0
        
        """
        Splitting the Targets (t) into Components -> (image, class), grid xy, grid wh, anchor
        """
        bc, gxy, gwh, a = t.chunk(4, 1)
        a = a.long().view(-1) # anchor index
        b, c = bc.T.long() # (image_id, class_id)
        gij = (gxy - offsets).long()
        gj, gi = gij.T

        return (
            (b, a, gj.clamp(0, pred_shape[2] - 1),  gi.clamp(0, pred_shape[3] - 1)),
            c,
            torch.cat((gxy - gij, gwh), -1),
            anchors[a]
        )


if __name__ == "__main__":
    anchors = torch.load('./assets/anchors.pt')
    pred = torch.load('./assets/preds.pt')
    targets = torch.load('./assets/targets.pt')

    compute_loss = Loss(anchors)


    loss, loss_items = compute_loss(pred[0], targets, anchors[0])

    print(loss)
    print(loss_items)