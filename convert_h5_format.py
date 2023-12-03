from PIL import Image
import numpy as np
import h5py
import argparse
import os


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f["rgb"])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f["depth"])
    return rgb, depth


def array_to_png(arr, h5_path, img_type):
    fn = os.path.splitext(os.path.basename(h5_path))[0]
    _, _, _, _, scene, frame = h5_path.split("/")
    frame_name = os.path.splitext(frame)[0]
    save_dir = os.path.join("../data/nyudepthv2/png", scene, img_type)
    save_path = os.path.join(save_dir, f"{frame_name}.png")

    img = Image.fromarray(arr.astype("uint8"))
    return img, save_path


def process_subdirectory(subdir):
    subdir_path = os.path.join("../data/nyudepthv2/train", subdir)
    for file_name in os.listdir(subdir_path):
        if file_name.endswith(".h5"):
            file_path = os.path.join(subdir_path, file_name)
            rgb_img, depth_img = h5_loader(file_path)

            # Save RGB image
            rgb_img, rgb_path = array_to_png(rgb_img, file_path, "rgb")
            os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
            rgb_img.save(rgb_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        metavar="DATA",
        default="",
        required=True,
        help="directory containing datasets of images in .h5 format.",
    )

    args = parser.parse_args()

    for subdir in os.listdir(args.data):
        subdir_path = os.path.join(args.data, subdir)
        if os.path.isdir(subdir_path):
            process_subdirectory(subdir)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--data",
#         metavar="DATA",
#         default="",
#         required=True,
#         help="directory containing datasets of images in .h5 format.",
#     )

#     args = parser.parse_args()

#     for file_name in os.listdir(args.data):
#         if file_name.endswith(".h5"):
#             file_path = os.path.join(args.data, file_name)
#             rgb_img, depth_img = h5_loader(file_path)

#             # Save RGB image
#             rgb_img, rgb_path = array_to_png(rgb_img, file_path, "rgb")
#             os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
#             rgb_img.save(rgb_path)


if __name__ == "__main__":
    main()
