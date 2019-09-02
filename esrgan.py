import re
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import argparse
import warnings
from pathlib import Path
from sys import exit


class SmartFormatter(argparse.HelpFormatter):
    """
         Custom Help Formatter used to split help text when '\n' was 
         inserted in it.
    """

    def _split_lines(self, text, width):
        r = []
        for t in text.splitlines():
            r.extend(argparse.HelpFormatter._split_lines(self, t, width))
            r.append("")
        return r


def enum_models(model_dir, aliases):
    models = {model.name: model for model in model_dir.rglob("*.pth")}
    for alias, original in aliases.items():
        models[alias] = model_dir / original
    return models


def get_models_help(models, aliases, model_docs):
    lines = []
    for model, docs in model_docs.items():
        if model not in models.keys():
            continue
        names = [model]
        for alias, original in aliases.items():
            if original == model:
                names.append(alias)
        quoted_names = (f'"{name}"' for name in names)
        lines.append(f"{' | '.join(quoted_names)}: {docs}")
    return "\n".join(lines)


def parse_args(models, models_help):
    parser = argparse.ArgumentParser(
        description="Upscale images with ESRGAN", formatter_class=SmartFormatter
    )
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
        help="The number of times to perform scaling. Defaults to 1.",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=models.keys(),
        default="0.8",
        help=f'The model to use for upscaling. Defaults to "0.8" (RRDB_PSNR_x4 - RRDB_ESRGAN_x4 x 0.8 interp).\n{models_help}',
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
    model_dir = Path(__file__).resolve().parent / "models"
    aliases = {
        "esrgan": "RRDB_ESRGAN_x4.pth",
        "psnr": "RRDB_PSNR_x4.pth",
        "0.8": "4x_interp_08.pth",
        "0.9": "4x_interp_09.pth",
        "desharpen": "1x_DeSharpen.pth",
        "jpeg20": "1x_JPEG_00-20.pth",
        "jpeg40": "1x_JPEG_20-40.pth",
        "jpeg60": "1x_JPEG_40-60.pth",
        "jpeg80": "1x_JPEG_60-80.pth",
        "jpeg100": "1x_JPEG_80-100.pth",
        "box": "4x_Box.pth",
        "misc": "4x_Misc_220000.pth",
        "facefocus": "4x_face_focus_275k.pth",
        "face": "4x_Faces_N_250000.pth",
        "fatality": "4x_Fatality_01_265000_G.pth",
        "unholy": "4x_unholy03.pth",
        "waifugan": "4x_WaifuGAN_v3_30000.pth",
        "manga109": "4x_Manga109Attempt.pth",
        "falcoon": "4x_falcoon300.pth",
        "rebout": "4x_rebout_325k.pth",
        "rebouti": "4x_rebout_interp.pth",
        "detoon": "4x_detoon_225k.pth",
        "detoon_alt": "4x_detoon_alt.pth",
        "bc1r": "1x_BC1_take2_260850.pth",
        "bc1f": "1x_BC1NoiseAgressiveTake3_400000_G.pth",
        "aa": "1x_Alias_200000_G.pth",
        "dedither": "1x_DEDITHER_32_512_126900_G.pth",
    }
    model_docs = {
        "RRDB_ESRGAN_x4.pth": "Official perceptual upscaling model.",
        "RRDB_PSNR_x4.pth": "Official PSNR upscaling model.",
        "4x_interp_08.pth": "RRDB_PSNR_x4 interpolated with RRDB_ESRGAN_x4 with 0.8 strength.",
        "4x_interp_09.pth": "RRDB_PSNR_x4 interpolated with RRDB_ESRGAN_x4 with 0.9 strength.",
        "4x_Box.pth": "General purpose upscaling. Larger dataset than RRDB_ESRGAN_x4.",
        "4x_Misc_220000.pth": "Surface upscaling. Works well as general/manga upscaler too.",
        "4x_Faces_N_250000.pth": "Face upscaling.",
        "4x_face_focus_275k.pth": "Face deblurring and upscaling.",
        "4x_Fatality_01_265000_G.pth": "Upscales pixel art.",
        "4x_rebout_325k.pth": "Upscales pixel art. Trained on KOF94 Rebout.",
        "4x_rebout_interp.pth": "Upscales pixel art. Trained on KOF94 Rebout. Interped.",
        "4x_falcoon300.pth": "Manga upscaling. Removes dithering.",
        "4x_unholy03.pth": "Manga upscaling. Interpolation of many models.",
        "4x_WaifuGAN_v3_30000.pth": "Manga upscaling.",
        "4x_Manga109Attempt.pth": "Manga upscaling.",
        "4x_ESRGAN_Skyrim_NonTiled_Alpha_NN_128_32_105000.pth": "Upscales greyscale maps. Trained on Skyrim textures.",
        "4x_detoon_225k.pth": "Tries to make toon images realistic.",
        "4x_detoon_alt.pth": "Tries to make toon images realistic. Softer version.",
        "1x_JPEG_00-20.pth": "Cleans up JPEG compression. For images with 0-20%% compression ratio.",
        "1x_JPEG_20-40.pth": "Cleans up JPEG compression. For images with 20-40%% compression ratio.",
        "1x_JPEG_40-60.pth": "Cleans up JPEG compression. For images with 40-60%% compression ratio.",
        "1x_JPEG_60-80.pth": "Cleans up JPEG compression. For images with 60-80%% compression ratio.",
        "1x_JPEG_80-100.pth": "Cleans up JPEG compression. For images with 80-100%% compression ratio.",
        "1x_BC1_take2_260850.pth": "Cleans up BC1 compression. Restricted version (only works with low-noise images).",
        "1x_BC1NoiseAgressiveTake3_400000_G.pth": "Cleans up BC1 compression. Free version (more aggressive than restricted).",
        "1x_cinepak_200000.pth": "Cleans up Cinepak, msvideo1 and Roq compression.",
        "1x_DeSharpen.pth": "Removes over-sharpening.",
        "1x_normals_generator_general_215k.pth": "Attempts to generate a normal map from a texture.",
        "1x_Alias_200000_G.pth": "Performs anti-aliasing on the image.",
        "1x_DEDITHER_32_512_126900_G.pth": "Tries to remove dithering patterns.",
    }
    models = enum_models(model_dir, aliases)
    models_help = get_models_help(models, aliases, model_docs)
    args = parse_args(models, models_help)
    model_path = model_dir / models[args.model]

    scale_pattern = re.compile('(\d+)x')
    scale_match = scale_pattern.match(model_path.stem)
    scale = int(scale_match.group(1)) if scale_match else 4

    device = torch.device(args.device)
    model = arch.RRDB_Net(
        3,
        3,
        64,
        23,
        gc=32,
        upscale=scale,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        res_scale=1,
        upsample_mode="upconv",
    )
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    for i, path in enumerate(
        Path(img_path)
        for img_glob in args.images
        for img_path in glob.glob(str(img_glob))
    ):
        print(i + 1, path.name)
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
    return 0


if __name__ == "__main__":
    exit(main())
