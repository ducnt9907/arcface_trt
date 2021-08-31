import argparse

import cv2
import numpy as np
import torch

from backbones import get_model
from numpy import dot
from numpy.linalg import norm
from datetime import datetime
import os
import struct
from torchsummary import summary
import time


@torch.no_grad()
def inference(weight, name, img, feat_path):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    img = torch.Tensor(img).cuda()
    net = get_model(name, fp16=False)
    net.cuda()
    # net.cpu()
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).data.cpu().numpy()

    f = open(feat_path, "r").read()
    a = f.split(' ')[:-1]
    a = np.asarray(a).astype('float')
    
    b = feat.reshape(512)
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    print("Cosine similarity: ", cos_sim)
    print("Diff: ", np.sum(abs(a-b)))
    print("Mean diff: ", np.sum(abs(a-b))/512)
    print("Max diff: ", np.max(abs(a-b)))

    print(feat[0][:10])
    print(a[:10])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--feat', type=str, default=None)
    args = parser.parse_args()
    inference(args.weight, args.network, args.img, args.feat)
