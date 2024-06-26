
import os
import json
import logging
import argparse

from pathlib import Path
from tifffile import imwrite
from datetime import datetime
from cellpose.io import imread
from cellpose import models, denoise

import numpy as np

now = datetime.now()
date_string = f'cellpose_{now.strftime("%Y-%m-%d_%H-%M-%S")}'

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help="Path to the image file.")
parser.add_argument('--model', type=str, required=True, help="Model type or path to a pretrained model.")
parser.add_argument('--denoise', action='store_true', help="Use a denoising model if specified.")
parser.add_argument('--output_dir', type=str, default=None, help="Changes default output dir. Defaults to image location.")

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def load_denoise_model(gpu: bool = True):
    return denoise.DenoiseModel(model_type="denoise_cyto3", gpu=gpu)


# model_type='cyto' or 'nuclei' or 'cyto2'
def load_model(model_identifier: str, gpu: bool = True, denoise_flag: bool = False):
    # If denoise_flag is true, then run a CellposeDenoiseModel
    if denoise_flag:
        if Path(model_identifier).is_file():
            return denoise.CellposeDenoiseModel(
                gpu=gpu, pretrained_model=model_identifier, restore_type="denoise_cyto3"
            )
        else:
            return denoise.CellposeDenoiseModel(
                gpu=gpu, model_type=model_identifier, restore_type="denoise_cyto3"
            )
    else:
        if Path(model_identifier).is_file():
            return models.CellposeModel(gpu=gpu, pretrained_model=model_identifier)
        else:
            return models.CellposeModel(gpu=gpu, model_type=model_identifier)


def run_predictions(model, image, channels):
    mask = None
    # Run CellposeDenoiseModel
    if isinstance(model, denoise.CellposeDenoiseModel):
        mask, _, _, image = model.eval(
            image,
            channels=channels,
            batch_size=256,

            diameter = 30,
            do_3D = True
        )

        print("Predictions Done")
        return mask

    # Run CellposeModel
    if isinstance(model, models.CellposeModel):
        mask, _, _ = model.eval(
            image,
            channels=channels,
            batch_size=256,

            diameter = None,
            do_3D = True,
            min_size = 2000,
            normalize = True,
        )

        print("Predictions Done")
        return mask


def main():
    # Load image info (currently will only support single images)
    image_path = Path(args.image)
    image_name = image_path.name
    image_directory = image_path.parent

    # Output directory structuring
    output_dir_name = args.output_dir if args.output_dir is not None else date_string
    output_dir = image_directory / output_dir_name
    os.makedirs(output_dir, exist_ok=True)

    # Load image and set channels
    channels = [[0, 0]]
    image = imread(image_path.as_posix())
    print(f'{image_name} has shape {np.shape(image)}')

    # If --denoise flag used, denoise and save image
    if args.denoise:
        denoise_model = load_denoise_model()
        denoised_image = denoise_model.eval([image], channels=channels, diameter=50.)
        imwrite(output_dir / f'denoised_{image_name}', denoised_image)
        print(f'Denoised image has shape {np.shape(denoised_image)}')

    # Run mask predictions and save image
    else:
        model = load_model(model_identifier=args.model, denoise_flag=False)
        mask = run_predictions(model, image, channels)
        imwrite(output_dir / f'mask_{image_name}', mask)
        print("Masks saved to", output_dir / f'mask_{image_name}')


if __name__ == '__main__':
    main()
