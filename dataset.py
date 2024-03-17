import os
import cv2
import math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
VALID_FOMAT = ['jpeg', 'jpg', 'png']

class Dataset:
    def __init__(self, path, mode, nc=20, imgsz=480, transform=None):
        self.path = path
        self.mode = mode
        self.img_size = imgsz
        self.transform = transform if transform else Compose([transforms.Resize((imgsz, imgsz)), transforms.ToTensor()])
        self.shapes = self.__checkfiles__()
        self.shapes = np.array(self.shapes)
        self.labels = [np.array(self.check_txtfile(label_file)) for label_file in self.labels_path]

    def __len__(self):
        return len(self.images)

    def __checkfiles__(self):
        NotFoundLabel = []
        CorruptedImage = []
        shapes = []
        self.images, self.labels_path =  [], []

        images = sorted(os.listdir(self.path + "/" + self.mode + "/images"))
        assert len(images), f"0 images are found in {self.path}"
        for index in range(len(images)):
            file = images[index]
            file_ext = os.path.basename(file).split('.')[-1]
            file_name = os.path.basename(file).split(f".{file_ext}")[0]
            image = os.path.join(self.path, self.mode, "images", file)

            im = cv2.imread(image)
            if im is None:
                CorruptedImage.append(image)
                print(f'Corrupted Image: {image}')
                continue
            
            h0, w0 = im.shape[:2]
            shapes.append((w0, h0))

            label = os.path.join(self.path, self.mode, "labels", file_name + '.txt')
            if not os.path.exists(label):
                warning = f"{label} not found\n"
                print(warning)
                NotFoundLabel.append(warning)
                continue
            
            self.images.append(image)
            self.labels_path.append(label)


        return shapes


    def load_image(self, i):
        """
        Loads an image by index, returning the image, its original dimensions, and resized dimensions.

        Returns (im, original hw, resized hw)
        """
        f = self.images[i]
        im = cv2.imread(f)  # BGR
        assert im is not None, f"Image Not Found {f}"

        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratioi
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

        return self.ims[i], (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

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
        label = self.labels_path[index]

        # image = Image.open(image).convert("RGB")

        image, h0w0, hw = self.load_image(index)
        image = Image.fromarray(image)
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
