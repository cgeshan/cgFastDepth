# sample run command
# python cgFastDepth/online.py --model ../results/mobilenet-nnconv5dw-skipadd-pruned.pth.tar --cam Examples/RGB-D/TUM2.yaml --folder sequences/rgbd_dataset_freiburg2_xyz/rgb --run

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

cudnn.benchmark = True

from PIL import Image
from datetime import datetime
from natsort import natsorted 
import cv2
import yaml
import time

import utils

args = utils.parse_command()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def parse_camera_config(config_path):
    try:
        with open(config_path, "r") as file:
            lines = file.readlines()[1:]
            yaml_content = "".join(lines)
            camera_config = yaml.safe_load(yaml_content)

            if "Camera.width" in camera_config and "Camera.height" in camera_config and "Camera.fps" in camera_config:
                camera_wid = camera_config["Camera.width"]
                camera_hei = camera_config["Camera.height"]
                camera_fps = camera_config["Camera.fps"]

                return camera_wid, camera_hei, camera_fps

            else:
                print("Error: Missing Camera.width or Camera.height in the YAML file.")

    except FileNotFoundError:
        print(f"Error: File not found - {config_path}")

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def run_single(model, image_path, camera_wid, camera_hei, is_folder=False):
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb_array = np.array(
        Image.fromarray(img).resize((224, 224), Image.BILINEAR)
    ).astype(np.double)
    rgb_array /= 255
    input = np.zeros([1, 3, 224, 224], dtype=np.float32)
    input[0, :, :, :] = np.transpose(rgb_array, (2, 0, 1))
    input = torch.from_numpy(input).to(device)
    result = model(input)
    output_img = np.squeeze(result.data.cpu().numpy()).copy()
    output_img_resized = cv2.resize(output_img, (camera_wid, camera_hei))
    return img, output_img_resized


def run_folder(model, folder_path, output_dir, camera_wid, camera_hei, camera_fps, fifo):
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)

    all_files = os.listdir(folder_path)
    image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sorted_files = natsorted(image_files, key=lambda x: datetime.utcfromtimestamp(float(x.split('.')[0])).strftime('%Y%m%d_%H%M%S.%f'))

    for filename in sorted_files:
        image_path = os.path.join(folder_path, filename)
        rgb, depth = run_single(model, image_path, camera_wid, camera_hei, is_folder=True)

        fifo.write(rgb.tobytes())
        fifo.write(depth.tobytes())

        time.sleep(1/camera_fps)

def run_from_txt(model, txt_file_path, camera_wid, camera_hei, fifo):
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)

    txt_file_path = f"{txt_file_path}timestamps.txt"

    with open(txt_file_path, 'r') as file:
        rgb_file_paths = file.read().splitlines()
        # print(rgb_file_paths)

    for rgb_file_path in rgb_file_paths:
        image_path = os.path.join(args.txt, f"rgb/{rgb_file_path}.png")
        rgb, depth = run_single(model, image_path, camera_wid, camera_hei, is_folder=True)

        fifo.write(rgb.tobytes())
        fifo.write(depth.tobytes())

def main():
    global args, output_dir

    camera_config_result = parse_camera_config(args.cam)

    if camera_config_result is not None:
        camera_wid, camera_hei, camera_fps = camera_config_result
        print(f"\nCamera width: {camera_wid}, Camera height: {camera_hei}")
    else:
        print("\nError: Unable to obtain camera configuration. Setting defaults...")
        camera_wid = 224
        camera_hei = 224
        camera_fps = 1
        print(f"Camera width: {camera_wid}, Camera height: {camera_hei}")

    print("\n## Attempting to load model...\n")
    if args.model:
        assert os.path.isfile(args.model), f"=> no model found at '{args.model}'"
        print(f"=> loading model '{args.model}'")
        checkpoint = torch.load(args.model)

        if type(checkpoint) is dict:
            args.start_epoch = checkpoint["epoch"]
            best_result = checkpoint["best_result"]
            model = checkpoint["model"]
            print(f"=> loaded best model (epoch {checkpoint['epoch']})")
            print("\n## Model loaded successfully...")
        else:
            model = checkpoint
            args.start_epoch = 0

        output_dir = os.path.dirname(args.model)

        fifo_path = "Custom/data_stream"
        if not os.path.exists(fifo_path):
            fifo_path = "../Custom/data_stream"
            if not os.path.exists(fifo_path):
                print("\n\n*** ERROR *** Cannot find named pipe, make sure it exists. Checked Custom/data_stream, ../Custom/data_stream")
                sys.exit(1)

        with open(fifo_path, "wb") as fifo:
            if args.run:
                if args.folder:
                    print(
                        "\n## Running depth estimation for online SLAM..."
                    )
                    run_folder(model, args.folder, output_dir, camera_wid, camera_hei, camera_fps, fifo)
                
                elif args.txt:
                    t0 = time.time()
                    print("\n## Running depth estimation for online SLAM using images from a txt file...")
                    run_from_txt(model, args.txt, camera_wid, camera_hei, fifo)
                    print(f"\n## Runtime: {time.time() - t0} seconds")
                
                terminate_signal = "terminate"
                fifo.write(terminate_signal.encode("utf-8"))
                print("terminate sent")

                fifo.close()
        return


if __name__ == "__main__":
    main()
