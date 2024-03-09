import torch


def box_iou(box1, box2, xywh=False, CIoU=False, eps=1e-6):
    

    (b1_x1, b1_y1, b1_x2, b1_y2), (b2_x1, b2_y1, b2_x2, b2_y2) = box1.chunk(4, -1), box2.chunk(4, -1)
    
    # if box format is x_center, y_center, width, height
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        (b1_x1, b1_y1, b1_x2, b1_y2) = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 +  h1 / 2
        (b2_x1, b2_y1, b2_x2, b2_y2) = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 +  h2 / 2
        
    w1, h1 = (b1_x2 - b1_x1), (b1_y2 - b1_y1).clamp(eps)
    w2, h2 = (b2_x2 - b2_x1), (b2_y2 - b2_y1).clamp(eps)

    inter = (b1_x2.minimum(b2_x2) - (b1_x1.maximum(b2_x1))).clamp(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # compute complete interaction overr union
    if CIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        c2 = (cw ** 2 +  ch ** 2).clamp(eps) # enclosed diogonal
        cd = (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y1 -  b1_y1 - b1_y2) ** 2 # cebter distance
        v = (4 / torch.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2) # aspect ration
        with torch.no_grad():
            alpha = v / ((1 - iou) + v)

        iou = iou - (cd / c2 + v * alpha)  

    return iou



if __name__ == "__main__":

    box1 = torch.tensor(
        [[50, 50, 100, 100]]
    )

    box2 = torch.tensor(
        [
            [60, 60, 110, 110],
            [66, 58, 120, 95]
        ]
    )

    iou = box_iou(box1, box2, CIoU=True)
    print(iou)