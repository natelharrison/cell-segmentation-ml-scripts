
import os
import json
import logging
import argparse

from pathlib import Path
from tifffile import imwrite
from datetime import datetime
from cellpose.io import imread
from cellpose import models, denoise

now = datetime.now()
date_string = f'cellpose_{now.strftime("%Y-%m-%d_%H-%M-%S")}'

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--denoise', action='store_true', default=False)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=date_string, help="Changes default output dir")

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


# model_type='cyto' or 'nuclei' or 'cyto2'
def load_model(model_identifier: str, gpu: bool = True, denoise_flag: bool = False):
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
    mask, _, _ = model.eval(
        image,
        channels=channels,
        batch_size=256,

        diameter = 60,
        do_3D = True,
        min_size = 2000,
        normalize = True,
        cellprob_threshold = -0.1
    )

    print("Predictions Done")
    return mask


def main():
    # Load model
    model = load_model(model_identifier=args.model, denoise_flag=args.denoise)

    # Load image info (currently will only support single images)
    image_path = Path(args.image)
    image_name = image_path.name

    # File structuring
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    channels = [[0, 0]]
    image = imread(image_path.as_posix())

    print("Computing Masks")
    mask = run_predictions(model, image, channels)
    print("Mask computed")

    imwrite(output_dir / f'mask_{image_name}', mask)


if __name__ == '__main__':
    main()
