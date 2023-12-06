# python local.py --model ../../results/mobilenet-nnconv5dw-skipadd-pruned.pth.tar
# --folder ../sequences/rgbd_dataset_freiburg1_room/rgb
# --cam ../Examples/RGB-D/TUM1.yaml --run

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

cudnn.benchmark = True

from PIL import Image
import cv2
import yaml

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

            if "Camera.width" in camera_config and "Camera.height" in camera_config:
                camera_wid = camera_config["Camera.width"]
                camera_hei = camera_config["Camera.height"]
                return camera_wid, camera_hei

            else:
                print("Error: Missing Camera.width or Camera.height in the YAML file.")

    except FileNotFoundError:
        print(f"Error: File not found - {config_path}")

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def rescale(depth, d_min=None, d_max=None):
    d_min = min(np.min(depth), np.min(depth))
    d_max = max(np.max(depth), np.max(depth))
    depth_relative = (depth - d_min) / (d_max - d_min)
    return depth_relative


def run_single(model, image_path, camera_wid, camera_hei, is_folder=False):
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb_array = np.array(
        Image.fromarray(img).resize((camera_wid, camera_hei), Image.BILINEAR)
    ).astype(np.double)
    rgb_array /= 255
    input = np.zeros([1, 3, camera_hei, camera_wid], dtype=np.float32)
    input[0, :, :, :] = np.transpose(rgb_array, (2, 0, 1))
    input = torch.from_numpy(input).to(device)
    result = model(input)
    output_img = np.squeeze(result.data.cpu().numpy())
    # print(output_img)
    output_img_rescaled = (output_img * (65536 / np.max(output_img))).astype(np.uint16)

    if not is_folder:
        plt.imshow(output_img)
        plt.show()

    file_name = os.path.splitext(os.path.basename(image_path))[0]

    if not is_folder:
        output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{file_name}_pred.png")

    else:
        output_dir = os.path.join(os.path.dirname(args.folder), "depth_preds")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{file_name}.png")

    if cv2.imwrite(save_path, output_img_rescaled):
        print(f"## Image successfully saved to {save_path}")
    else:
        print("*** Error *** Image not saved...")


def run_folder(model, folder_path, output_dir, camera_wid, camera_hei):
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            print(image_path)
            run_single(model, image_path, camera_wid, camera_hei, is_folder=True)


def main():
    global args, output_dir

    camera_config_result = parse_camera_config(args.cam)

    if camera_config_result is not None:
        camera_wid, camera_hei = camera_config_result
        print(f"Camera width: {camera_wid}, Camera height: {camera_hei}")
    else:
        print("Error: Unable to obtain camera configuration. Setting defaults...")
        camera_wid = 224
        camera_hei = 224
        print(f"Camera width: {camera_wid}, Camera height: {camera_hei}")
        # return

    print("\n## Attempting to load model...\n\n")
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

        if args.run:
            if args.image:
                print("\n## Running depth estimation on image...")
                run_single(model, args.image, camera_wid, camera_hei)

            elif args.folder:
                print(
                    "\n## Running depth estimation on images in the specified folder..."
                )
                run_folder(model, args.folder, output_dir, camera_wid, camera_hei)
        return


if __name__ == "__main__":
    main()
