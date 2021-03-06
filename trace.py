import os
import cv2
import numpy as np
import torch
import architecture as arch
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert models to TorchScript")
    parser.add_argument(
        "models",
        nargs="*",
        type=Path,
        help="The models to process. Defaults to all in models directory.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=False,
        help="The directory to write output to. Defaults to jit_models in ESRGAN directory.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="The device to use for upscaling. Defaults to cuda.",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        default="_jit.pth",
        help="The suffix to add to output model name.",
    )
    parser.add_argument("-f", "--force", help="Whether to overwrite existing files.")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    model_dir = script_dir / "models"
    out_dir = args.out_dir if args.out_dir else script_dir / "jit_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        models = args.models
    else:
        models = model_dir.rglob("*.pth")

    device = torch.device(args.device)

    # read image
    img_path = script_dir / "LR/baboon.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    for model_path in models:
        if not model_path.is_file():
            print(f"{str(model_path)} does not exist, skipping...")
            continue

        out_path = out_dir / (model_path.stem + args.suffix)
        if not args.force and out_path.is_file():
            print(f"{str(out_path)} already exists, skipping...")
            continue

        print(f"Tracing: {str(model_path)}")

        state_dict = torch.load(model_path)
        if "conv_first.weight" in state_dict:
            print("Error: Attempted to load a new-format model")
            return 1

        # Extract model information
        scale2 = 0
        max_part = 0
        in_nc = 3
        out_nc = 3
        nf = 64
        nb = 23
        for part in list(state_dict):
            parts = part.split(".")
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == "sub":
                nb = int(parts[3])
            elif n_parts == 3:
                part_num = int(parts[1])
                if part_num > 6 and parts[2] == "weight":
                    scale2 += 1
                if part_num > max_part:
                    max_part = part_num
                    out_nc = state_dict[part].shape[0]
        upscale = 2 ** scale2
        in_nc = state_dict["model.0.weight"].shape[1]
        nf = state_dict["model.0.weight"].shape[0]

        device = torch.device(args.device)
        net = arch.RRDB_Net(
            in_nc,
            out_nc,
            nf,
            nb,
            gc=32,
            upscale=upscale,
            norm_type=None,
            act_type="leakyrelu",
            mode="CNA",
            res_scale=1,
            upsample_mode="upconv",
        )
        net.load_state_dict(state_dict, strict=True)
        del state_dict
        net.eval()

        for k, v in net.named_parameters():
            v.requires_grad = False
        net = net.to(device)

        with torch.jit.optimized_execution(should_optimize=True):
            # traced_script_module = torch.jit.trace(net, img_LR)
            traced_script_module = torch.jit.script(net)
            print(f"Saving to: {str(out_path)}")
            try:
                with out_path.open("wb") as out_file:
                    torch.jit.save(traced_script_module, out_file)
            except:
                os.remove(out_path)
                raise


if __name__ == "__main__":
    main()
