import os
import time
import csv
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

cudnn.benchmark = True

from metrics import AverageMeter, Result
import models
import utils

args = utils.parse_command()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def run(model, epoch):
    model.eval()

    pass


def main():
    global args, output_dir

    # load the model
    print("\n\n## Attempting to load model...")
    if args.run:
        assert os.path.isfile(args.run), "=> no model found at '{}'".format(args.run)
        print("=> loading model '{}'".format(args.run))
        checkpoint = torch.load(args.run)

        if type(checkpoint) is dict:
            args.start_epoch = checkpoint["epoch"]
            best_result = checkpoint["best_result"]
            model = checkpoint["model"]
            print("=> loaded best model (epoch {})".format(checkpoint["epoch"]))
        else:
            model = checkpoint
            args.start_epoch = 0

        output_dir = os.path.dirname(args.run)

        run(model, args.start_epoch)
        return
