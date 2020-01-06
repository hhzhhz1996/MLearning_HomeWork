from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable




class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.path = folder_path
        self.names = os.listdir(folder_path)
        self.img_size = img_size

    def __getitem__(self, index):
        path = self.path + self.names[index % len(self.names)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return self.names[index % len(self.names)], img

    def __len__(self):
        return len(self.names)

def test(img_path, anno_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)
    
    
    model.load_state_dict(torch.load("weights/params.pth"))
    model.eval()
    dataloader = DataLoader(
        ImageFolder(folder_path=img_path, img_size=416),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    classes = load_classes('data/classes.names')  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 0.01, 0.5)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
        

    # Bounding-box colors
    f = open(anno_path+'det_test_带电芯充电宝.txt', 'w')
    g = open(anno_path+'det_test_不带电芯充电宝.txt', 'w')

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        img = np.array(Image.open(img_path + path))
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, 416, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if cls_pred == 1:
                    f.write(path[:-4] + " {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}".format(conf, x1, y1, x2, y2) + '\n')
                else:
                    g.write(path[:-4] + " {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}".format(conf, x1, y1, x2, y2) + '\n')
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

    f.close()
    g.close()



if __name__ == "__main__":
    
    img_path = 'c:/users/86132/desktop/map/Image_test/'
    anno_path = '../predicted_file/'
    
    test(img_path, anno_path)