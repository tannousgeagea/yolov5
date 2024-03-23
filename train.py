import torch
import torch.nn as nn
from tqdm import tqdm

from loss import Loss
from dataset import Dataset
from model import YOLOv5
from utils import save_model, load_model
from metrics import non_maximum_suppression, mean_average_precision

anchors = torch.load('./assets/anchors.pt')
S = torch.tensor([8, 16, 32]).view(-1, 1, 1)
print(anchors)

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
    losses = []

    model.train()
    loss_dict = {}
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            p = model(x)
            loss = (
                loss_fn(p[0], y, anchors[0])[0] + 
                loss_fn(p[1], y, anchors[1])[0] + 
                loss_fn(p[2], y, anchors[2])[0]
            )
            
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        lr = scheduler.get_last_lr()[0]
        
        pbar.set_postfix(
            epoch=f"{epoch + 1}", 
            loss=f"{mean_loss:.2f}", 
            gpu_usage=f"{current_memory:.2f} Gb", 
            max_gpu_usage=f"{max_memory:.2f} Gb", 
            lr=f"{lr}"
            )
        


data_path = '/home/appuser/data/animals.v2-release.yolov5pytorch'
train_dataset = Dataset(
    path=data_path,
    mode='train',
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=train_dataset.collate_fn
    )


val_dataset = Dataset(
    path=data_path,
    mode='valid',
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=16, 
    shuffle=False, 
    collate_fn=train_dataset.collate_fn
    )

# initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLOv5(in_channels=3).to(device)
load_model(model, checkpoint_file='./assets/yolov5.pt')


# initialize loss
loss_fn = Loss()

# initialze optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001, weight_decay=1e-4,
)

scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90], gamma=0.1)

mean_ap = 0.

epochs = 100

anchors = anchors.to(device)
for epoch in range(epochs):
    train_step(
        model=model,
        dataloader=train_dataloader,
        anchors=anchors,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        epoch=epoch,
        device=device
    )
    
    scheduler.step()
    
    checkpoint = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    
    
    save_model(model=model, save_path='./assets/yolov5.pt', checkpoint=checkpoint)
    
    





