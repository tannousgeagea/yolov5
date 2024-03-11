import torch
import torch.nn as nn
from tqdm import tqdm

from loss import Loss
from dataset import Dataset
from model import YOLOv5
from metrics import non_maximum_suppression, mean_average_precision

def train_step(
    model, 
    dataloader, 
    anchors,
    loss_fn, 
    optimizer, 
    scaler,
    scheduler,
    epoch=0, 
    device='cuda',
):  
    pbar = tqdm(dataloader, ncols=125)
    losses = 0.0

    model.train()
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            p = model(x)
            for i, pi in enumerate(p):
                loss, loss_items = loss_fn(pi, y, anchors[i])
                losses += loss
            
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = losses / i
        current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(epoch=f"{epoch + 1} / {EPOCHS}", loss=f"{mean_loss:.2f}", gpu_usage=f"{current_memory:.2f} Gb", max_gpu_usage=f"{max_memory:.2f} Gb", lr=f"{lr}")


