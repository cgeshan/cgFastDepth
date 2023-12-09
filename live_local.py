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


def run_single(model, frame, camera_wid, camera_hei, is_folder=False):
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)

    frame = cv2.resize(frame, (camera_wid, camera_hei))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb_array = np.array(img).astype(np.double) / 255.0
    input = np.zeros([1, 3, camera_hei, camera_wid], dtype=np.float32)
    input[0, :, :, :] = np.transpose(rgb_array, (2, 0, 1))
    input = torch.from_numpy(input).to(device)

    start_time = time.time()
    result = model(input)
    elapsed_time = time.time() - start_time

    output_img = np.squeeze(result.data.cpu().numpy())
    output_img_rescaled = rescale(output_img)

    fps = 1.0 / elapsed_time
    fps_text = f"FPS: {fps:.2f}"

    cv2.putText(
        output_img_rescaled,
        fps_text,
        (10, 30),  
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),  
        2,
        cv2.LINE_AA,
    )

    return output_img_rescaled, fps


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

        cap = cv2.VideoCapture(0)

        # Get the webcam's resolution
        default_camera_wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_camera_hei = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_time = time.time()
        fps_sum = 0.0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame from webcam.")
                break

            estimation, fps = run_single(
                model, frame, default_camera_wid, default_camera_hei, is_folder=True
            )

            fps_sum += fps
            frame_count += 1

            avg_fps = fps_sum / frame_count
            avg_fps_text = f"Avg FPS: {avg_fps:.2f}"

            resolution_text = f"Resolution: {default_camera_wid} x {default_camera_hei}"

            cv2.putText(
                estimation,
                avg_fps_text,
                (10, 60),  # Adjusted text position
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                estimation,
                resolution_text,
                (10, 90),  # Adjusted text position
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Webcam Feed", frame)
            cv2.imshow("Depth Estimation", estimation)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
