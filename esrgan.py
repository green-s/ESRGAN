import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import argparse
import warnings
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Upscale images with ESRGAN")
    parser.add_argument("images", nargs="+", type=Path, help="The images to process.")
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=False,
        help="The directory to write output to. Defaults to source directory.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=1,
        help="The number of times to repeat scaling (by x4). Defaults to 1.",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["esrgan", "psnr", "0.8", "0.9"],
        default="0.8",
        help="The model to use for upscaling. Defaults to RRDB_ESRGAN_x4",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="The device to use for upscaling. Defaults to cuda.",
    )
    parser.add_argument(
        "-e",
        "--end",
        default="_scaled",
        help="The suffix to append to scaled images. Defaults to `_scaled`.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model_map = {
        "esrgan": "RRDB_ESRGAN_x4.pth",
        "psnr": "RRDB_PSNR_x4.pth",
        "0.8": "interp_08.pth",
        "0.9": "interp_09.pth",
    }
    model_path = Path(__file__).resolve().parent / "models" / model_map[args.model]

    model = arch.RRDB_Net(
        3,
        3,
        64,
        23,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        res_scale=1,
        upsample_mode="upconv",
    )
    model.load_state_dict(torch.load(str(model_path)), strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    for i, path in enumerate(
        Path(img_path)
        for img_glob in args.images
        for img_path in glob.glob(str(img_glob))
    ):
        print(i + 1, path.stem)
        # read image
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = img * 1.0 / 255

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(args.scale):
                img = torch.from_numpy(
                    np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
                ).float()
                img_LR = img.unsqueeze(0)
                img_LR = img_LR.to(device)

                img = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))

        img = (img * 255.0).round()

        out_dir = args.out_dir if args.out_dir is not None else path.parent
        out_path = out_dir / (path.stem + args.end + ".png")
        cv2.imwrite(str(out_path), img)

    print("Done")


if __name__ == "__main__":
    main()
