import os
import cv2
import time
import torch
import shutil
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def iou_width_height(boxes1, boxes2):
    """
    Calculate the IoU of two sets of boxes that are in the center-size notation.
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    return intersection / union


def convert_trueboxes(true_boxes, S=13):
    """
    Convert YOLO model label to original label format.

        Args:
        true_boxes (torch.Tensor): true boxes (N, num_anchors, S, S, 6),
                                    where S is the grid size, and C is the number of classes.
                                    The last 5 elements in the tensor are the objectness score,
                                    and the normalized bounding box coordinates [x, y, w, h].
        anchors (List): the anchors used for the predictions [3]
        S (int): The size of the grid (number of cells in one dimension).
        C (int): The number of classes.

        Returns:
        list: A list of true boxes, where each box is represented as a list:
            [class_id, x_center, y_center, width, height].
            The coordinates are normalized with respect to the image size.
    """

    batch_size = true_boxes.shape[0]
    score = true_boxes[..., 0:1]
    boxes = true_boxes[..., 1:5]
    best_class = true_boxes[..., 5:6]

    cell_indices = torch.arange(S).repeat(batch_size, 3, S, 1).unsqueeze(-1)
    x = 1 / S * (boxes[..., 0:1] + cell_indices)
    y = 1 / S * (boxes[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * boxes[..., 2:4]
    converted_boxes = torch.cat([best_class, score, x, y, w_h], dim=-1).reshape(batch_size, 3 * S * S, 6)
    return converted_boxes.tolist()

def convert_predboxes(predictions,anchors, S=13):
    """
    Convert YOLO model prediction to readable format.

        Args:
        predictions (torch.Tensor): true boxes (N, num_anchors, S, S, 4 + 1 + C),
                                    where S is the grid size, and C is the number of classes.
                                    The fisrt 5 elements in the tensor are the objectness score,
                                    and the normalized bounding box coordinates [x, y, w, h].
        anchors (List): the anchors used for the predictions [3]
        S (int): The size of the grid (number of cells in one dimension).
        C (int): The number of classes.

        Returns:
        list: A list of boxes, where each box is represented as a list:
            [class_id, confidence, x_center, y_center, width, height].
            The coordinates are normalized with respect to the image size.
    """
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    best_class = predictions[..., 5:]
    boxes = predictions[..., 1:5]
    scores = predictions[..., 0:1]

    anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
    boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2])
    boxes[..., 2:] = torch.exp(boxes[..., 2:]) * anchors
    scores = torch.sigmoid(scores)
    best_class = torch.argmax(best_class, dim=-1).unsqueeze(-1)

    device = predictions.device
    cell_indices = torch.arange(S).repeat(batch_size, 3, S, 1).unsqueeze(-1).to(device)
    x = 1 / S * (boxes[..., 0:1] + cell_indices)
    y = 1 / S * (boxes[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * boxes[..., 2:4]
    converted_boxes = torch.cat([best_class, scores, x, y, w_h], dim=-1).reshape(batch_size, num_anchors * S * S, 6)
    return converted_boxes.tolist()

def draw_bounding_boxes(image, boxes):
    """
    Draws bounding boxes on the image.
    
    Args:
    - image: The image on which to draw, as a NumPy array.
    - boxes: A list of boxes, each box is a list of [x_center, y_center, width, height],
             with values normalized between 0 and 1.
    
    Returns:
    - The image with bounding boxes drawn on it.
    """
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    for box in boxes:

        _, _, x_center, y_center, box_width, box_height = box
        xmin, ymin, xmax, ymax = (
            int((x_center - box_width / 2) * width),
            int((y_center - box_height / 2) * height),
            int((x_center + box_width / 2) * width),
            int((y_center + box_height / 2) * height),
        )        
        
        # Draw the rectangle on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    return image

def pull(csv_file, source='', output='./data'):
    if not os.path.exists(csv_file):
        logging.warning('csv_file %s does not exist' %csv_file)
        return None

    annotations = pd.read_csv(csv_file)
    mode = os.path.basename(csv_file).split('.csv')[0]

    pbar = tqdm(range(len(annotations)), ncols=100)
    for index in pbar:
        label_path = os.path.join(source, 'labels', annotations.iloc[index, 1])
        image_path = os.path.join(source, 'images', annotations.iloc[index, 0])

        image_exists = os.path.exists(image_path)
        label_exists = os.path.exists(label_path)


        if image_exists and label_exists:

            if not os.path.exists(output + "/" + mode + "/images/"):
                os.makedirs(output + "/" + mode + "/images/")

            shutil.copy(image_path, output + "/" + mode + "/images/")

            if not os.path.exists(output + "/" + mode + "/labels/"):
                os.makedirs(output + "/" + mode + "/labels/")

            shutil.copy(label_path, output + "/" + mode +  "/labels")
        else:
            pbar.write('Image or Label does not exis: %s' %label_path)

    
def save_model(model, save_path, checkpoint=None):
    """
    Saves the PyTorch model to the specified path.

    Parameters:
    model (torch.nn.Module): The trained model to be saved.
    save_path (str): The file path where the model should be saved.
    checkpoint (dict, optional): A dictionary of additional information to save, 
                                 e.g., optimizer state, current epoch, loss.
    """
    if checkpoint is not None:
        # If there's additional information to be saved along with the model
        state = {
            'state_dict': model.state_dict(),
            **checkpoint
        }
    else:
        # If only the model needs to be saved
        state = model.state_dict()

    torch.save(state, save_path)

def load_model(model, checkpoint_file, optimizer=None):
    """
    Loads the PyTorch model and optionally its optimizer from the specified path.

    Parameters:
    model (torch.nn.Module): The model instance for which the state dict will be loaded.
    checkpoint_file (str): The file path from where the model should be loaded.
    optimizer (torch.optim.Optimizer, optional): The optimizer instance for which the state dict will be loaded, if available.

    Returns:
    dict or None: Returns the additional checkpoint data (like epoch, loss, etc.) if available, otherwise None.
    """
    checkpoint = torch.load(checkpoint_file)

    # Load model state
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if optimizer is provided and state is available
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"Model loaded from {checkpoint_file}")

    # return {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'optimizer_state_dict']}

def log_training_progress(metrics, speed, end='\r'):
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    print(
        f"{metrics['epoch']:10}"
        f"{gpu_mem:10}"
        f"{metrics['box_loss']:10}"
        f"{metrics['cls_loss']:10}"
        f"{metrics['obj_loss']:10}"
        f"{metrics['instances']:10}"
        f"{metrics['precision']:10}"
        f"{metrics['recall']:10}"
        f"{metrics['mAP50']:10}"
        f"{metrics['mAP50_95']:10}"
        f"{metrics['size']} ", end=end, flush=True)


if __name__ == "__main__":
    # csv_file = "/home/appuser/data/PASCAL_VOC/100examples.csv"
    # pull(csv_file, source="/home/appuser/data/PASCAL_VOC/train", output="/home/appuser/data")

    # Simulated call to the logging function
    # Example dynamic logging simulation


    header = "Epoch | GPU Mem | Box Loss | Class Loss | Obj Loss | Instances | Precision | Recall | mAP50 | mAP50-95"
    header = header.split(' | ')
    line= ''
    for h in header:
        line += f"{h:10}"

    for epoch in range(10):
        print(line)
        for batch in range(1, 101):
            # Example metrics that might be changing each batch
            metrics = {
                'epoch': f"{epoch}",
                'box_loss': 1.0 - 0.01 * batch,  # Example decreasing box_loss
                'cls_loss': 0.5 + 0.01 * batch,  # Example increasing cls_loss
                'obj_loss': 0.3,  # Example constant obj_loss
                'instances': 32,  # Example constant instance count
                'precision': 0.6,  # Example constant precision
                'recall': 0.8,  # Example constant recall
                'mAP50': 0.5,  # Example constant mAP50
                'mAP50_95': 0.4,  # Example constant mAP50-95
                'size': '640: 100%'  # Example constant size
            }
            speed = 5.0  # Example constant speed
            epoch = 1
            total_epochs = 3

            log_training_progress(metrics, speed)

            # Simulate time delay of a training batch
            time.sleep(0.1)

        print('')
        
        
def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y