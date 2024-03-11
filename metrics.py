import torch
from collections import Counter

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

def non_maximum_suppression(
    boxes, 
    iou_threshold=0.5, 
    conf=0.25,
    xywh=False
    ):
    """
    Perform Non-Maximum Suppression (NMS) on the bounding boxes.

    Args:
    boxes (numpy.ndarray): A 2D array of bounding boxes, where each row is [class_id, score, x1, y1, x2, y2].
    threshold (float): The IoU (Intersection over Union) threshold used to compare the bounding boxes.

    Returns:
    list: A list of the bounding boxes to keep.

    Description:
    This function takes a list of bounding boxes, each with a class ID and a confidence score,
    and eliminates boxes that have a high overlap (as determined by the IoU threshold).
    It processes each class separately and keeps only the box with the highest score
    in case of high IoU overlap with other boxes of the same class.
    """

    keep = [box for box in boxes if box[1] > conf]
    keep = sorted(keep, key=lambda x: x[1], reverse=True)
    boxes_after_nms = []
    while keep:
        chosen_box = keep.pop(0)
        keep = [
            box
            for box in keep
            if box[0] != chosen_box[0]
            or iou(
                torch.tensor(box[2:]).unsqueeze(0),
                torch.tensor(chosen_box[2:]).unsqueeze(0),
                xywh=xywh
            ) < iou_threshold
        ]

        boxes_after_nms.append(chosen_box)
    
    return boxes_after_nms


def mean_average_precision(predictions, target, iou_threshold=0.5, num_classes=20):
    """
    Compute the mean Average Precision (mAP) for object detection.

    Args:
    predictions (list of lists): A nested list where each sublist contains details of a detected box in the format 
                                    [image_id, class_id, confidence_score, x1, y1, x2, y2].
    ground_truth_boxes (list of lists): A nested list where each sublist contains details of a ground truth box in the format 
                                        [image_id, class_id, x1, y1, x2, y2].
    iou_threshold (float): The IoU threshold used to determine whether a detection is a true positive or a false positive.
    num_classes: int indicating the number of classes

    Returns:
    float: The mean Average Precision (mAP) across all classes.

    Description:
    This function calculates the mean Average Precision (mAP), a common metric used for evaluating the accuracy of object detectors.
    It involves the following steps:
    1. For each class, match detected boxes to ground truth boxes using the IoU threshold. A detected box is considered a true positive 
       if its IoU with a ground truth box is above the threshold, and there hasn't been another detection with higher confidence for 
       the same ground truth box.
    2. Calculate precision and recall for each class at various detection confidence thresholds.
    3. Average the precision at each recall level across all classes.
    
    Note: This function assumes that the boxes within each list are sorted by confidence scores in descending order.
    """

    average_precisions = []
    results =  {}
    epsilon = 1e-6

    for class_id in range(num_classes):
        # take only detections that belong to the class index
        detections_per_class = [detection for detection in predictions if detection[1] == class_id]
        target_per_class = [gt for gt in target if  gt[1] == class_id]

        # find the amount of boxes that belong to this class per image {image_id: number of boxes}
        amount_boxes_per_image = Counter([gt[0] for gt in target_per_class])
        for k, v in amount_boxes_per_image.items():
            # e.g. {0: (0, 0, 0), 1: (0, 0), ...}
            amount_boxes_per_image[k] = torch.zeros(v)
        # sort detections of the current class by confidence score
        detections_per_class.sort(key=lambda x: x[2], reverse=True)

        # initialize zeros for true positive and false positive
        TP = torch.zeros((len(detections_per_class)))
        FP = torch.zeros((len(detections_per_class)))
        total_true_boxes = len(target_per_class)
        
        # iterate over all prediction of this class_id
        for pred_idx, pred in enumerate(detections_per_class):
            best_iou = 0
            best_box = -1

            # check only true bounding boxes that have same image id
            gt_per_image = [gt for gt in target_per_class if gt[0] == pred[0]]
            for gt_idx, gt in enumerate(gt_per_image):
                box_iou = iou(
                    torch.tensor(pred[3:]).unsqueeze(0),
                    torch.tensor(gt[3:]).unsqueeze(0)
                )

                # check best iou 
                if box_iou > best_iou:
                    best_iou = box_iou
                    best_box = gt_idx
            
            # check true positive and false positive
            if best_iou > iou_threshold:
                # if the true box has  not been  registered yet or counted
                if amount_boxes_per_image[pred[0]][best_box] == 0:
                    # count one true positive
                    TP[pred_idx] = 1
                    # register true box to avoid duplication
                    amount_boxes_per_image[pred[0]][best_box] == 1
                else:
                    # the true box was already counted earlier
                    FP[pred_idx] = 1
            else:
                # best iou  less than threshold
                FP[pred_idx] = 1
        
        # sum true positive and false positive cumulatively 
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        
        # recall =  TP / (TP + FN) -> TP / total number of true boxes
        recall = TP_cumsum / (total_true_boxes + epsilon)

        #  precision = TP / (TP + FP)
        precision = torch.divide( TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        # we need to start from point(0 , 1)
        precision = torch.cat((torch.tensor([1]), precision))
        recall = torch.cat((torch.tensor([0]), recall))
        average_precision = torch.trapz(precision, recall)

        if not str(class_id) in results.keys():
            results[str(class_id)] = {}

        results[str(class_id)] =  {
            'tp': TP,
            'fp': FP,
            'total_true_boxes': total_true_boxes,
            'precision': precision,
            'recall': recall,
            'average_precision': average_precision
        } 

        average_precisions.append(average_precision)
    
    return sum(average_precisions) / len(average_precisions), results

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