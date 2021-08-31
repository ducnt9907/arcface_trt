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
def inference(weight, name, save_path):
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    
    print(net)
    f = open(save_path, 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    inference(args.weight, args.network, args.save_path)
