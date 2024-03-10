import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
VALID_FOMAT = ['jpeg', 'jpg', 'png']

class Dataset:
    def __init__(self, path, mode, nc=20, transform=None):
        self.path = path
        self.mode = mode
        self.transform = transform if transform else Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
        self.__checkfiles__()

    def __len__(self):
        return len(self.images)

    def __checkfiles__(self):
        NotFoundLabel = []
        self.images = []
        self.labels =  []
        images = sorted(os.listdir(self.path + "/" + self.mode + "/images"))
        for index in range(len(images)):
            file = images[index]
            file_ext = os.path.basename(file).split('.')[-1]
            file_name = os.path.basename(file).split(f".{file_ext}")[0]
            image = os.path.join(self.path, self.mode, "images", file)
            label = os.path.join(self.path, self.mode, "labels", file_name + '.txt')
            if not os.path.exists(label):
                warning = f"{label} not found\n"
                NotFoundLabel.append(warning)
                continue
            
            self.images.append(image)
            self.labels.append(label)

    def check_txtfile(self, txtfile):
        # check if txtfile exist
        boxes = []
        if not os.path.exists(txtfile):
            return boxes

        with open(txtfile, 'r') as f:
            for label in f.readlines():
                class_id, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_id, x, y, w, h])
        
        return boxes  

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = Image.open(image).convert("RGB")
        boxes = self.check_txtfile(label)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        nt = len(boxes)
        targets = torch.zeros((nt, 6))
        targets[:, 1:] = boxes
        return image, targets

    @staticmethod
    def collate_fn(batch):
        """Batches images, labels, paths, and shapes, assigning unique indices to targets in merged label tensor."""
        im, label = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes):
        for transform in self.transforms:
            image, bboxes = transform(image), bboxes
        return image, bboxes

if __name__ == "__main__":
    transform = Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    dataset = Dataset(path='/home/appuser/data/', mode='train', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)

    train_fea, train_labels = next(iter(dataloader))

    print(f"Train Feature: {train_fea.shape}")
    print(f"Train Targets: {train_labels.shape}")
