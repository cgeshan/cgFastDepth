import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from torchvision import transforms

import cv2

from PIL import Image

cudnn.benchmark = True

from metrics import AverageMeter, Result
import models
import utils

args = utils.parse_command()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def rescale(depth, d_min=None, d_max=None):
    d_min = min(np.min(depth), np.min(depth))
    d_max = max(np.max(depth), np.max(depth))
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * (depth_relative)  # H, W, C


def run_single(model, image_path):
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_array = np.array(Image.fromarray(img).resize((224, 224), Image.BILINEAR)).astype(np.double)
    rgb_array /= 255
    x = np.zeros([1, 3, 224, 224], dtype=np.float32)
    x[0, :, :, :] = np.transpose(rgb_array, (2, 0, 1))
    x = torch.from_numpy(x).to(device)
    result = model(x)
    output_img = np.squeeze(result.data.cpu().numpy())

    plt.imshow(output_img)
    plt.show()

    output_img_rescaled = rescale(output_img)

    output_dir = os.path.join(os.path.dirname(args.image), "depth_preds")
    os.makedirs(output_dir, exist_ok=True)

    # Create the save path in the "depth_preds" directory
    file_name = os.path.splitext(os.path.basename(args.image))[0]
    save_path = os.path.join(output_dir, file_name)

    if cv2.imwrite(save_path + ".png", output_img_rescaled):
        print(f"## Image successfully saved to {save_path}.png")
    else:
        print("*** Error *** Image not saved...")

def run_folder(model, image_path, output_dir):
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_array = np.array(Image.fromarray(img).resize((224, 224), Image.BILINEAR)).astype(np.double)
    rgb_array /= 255
    x = np.zeros([1, 3, 224, 224], dtype=np.float32)
    x[0, :, :, :] = np.transpose(rgb_array, (2, 0, 1))
    x = torch.from_numpy(x).to(device)
    result = model(x)
    output_img = result.detach().cpu().numpy()
    output_img = np.squeeze(output_img)

    # plt.imshow(output_img)
    # plt.show()

    output_img_rescaled = rescale(output_img)

    output_dir = os.path.join(os.path.dirname(args.folder), "depth_preds")
    os.makedirs(output_dir, exist_ok=True)

    # Create the save path in the "depth_preds" directory
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, file_name)

    if cv2.imwrite(save_path + ".png", output_img_rescaled):
        print(f"## Image successfully saved to {save_path}.png")
    else:
        print("*** Error *** Image not saved...")


def main():
    global args, output_dir

    # load the model
    print("\n## Attempting to load model...\n\n")
    if args.model:
        assert os.path.isfile(args.model), "=> no model found at '{}'".format(args.model)
        print("=> loading model '{}'".format(args.model))
        checkpoint = torch.load(args.model)

        if type(checkpoint) is dict:
            args.start_epoch = checkpoint["epoch"]
            best_result = checkpoint["best_result"]
            model = checkpoint["model"]
            print("=> loaded best model (epoch {})".format(checkpoint["epoch"]))
            print("\n## Model loaded successfully...")
        else:
            model = checkpoint
            args.start_epoch = 0

        output_dir = os.path.dirname(args.model)

        if args.run:
            if args.image:
                print("\n## Running depth estimation on image...")
                run_single(model, args.image)

            elif args.folder:
                print("\n## Running depth estimation on images in the specified folder...")
                for filename in os.listdir(args.folder):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        image_path = os.path.join(args.folder, filename)
                        print(image_path)
                        run_folder(model, image_path, output_dir)
        return
    

if __name__ == "__main__":
    main()

