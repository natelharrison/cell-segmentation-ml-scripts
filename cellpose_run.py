import os
import argparse
import json
from pathlib import Path
import logging
from cellpose import models, io
from cellpose.io import imread
from datetime import datetime

now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--image_path', type=str, default='')
parser.add_argument('--model', type=str, default='cyto2')
parser.add_argument('--kwargs', type=str,
    default='{"diameter": 30, "do_3D": true, "resample": true, "min_size": 5000, "augment": true}')
parser.add_argument('--save_name', type=str, default=date_string)
parser.add_argument('--batch_num', type=str, default=None)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


# model_type='cyto' or 'nuclei' or 'cyto2'
def load_model(
        model_path: Path,
        gpu: bool = True
) -> models.CellposeModel:

    return models.CellposeModel(gpu=gpu, pretrained_model=model_path.as_posix())


def main():
    model_path = Path(args.model)
    model = load_model(model_path)

    image_path = Path(args.image_path)
    image_name = image_path.stem

    save_name = args.save_name
    batch_num = args.batch_num

    save_dir = image_path.parent / save_name
    if batch_num is not None:
        save_dir = save_dir / model_path.stem
        image_name = f"{batch_num}_{image_name}"
    os.makedirs(save_dir, exist_ok=True)

    channels = [[0,0]]
    image = imread(image_path.as_posix())

    try:
        kwargs = json.loads(args.kwargs)
    except ValueError as e:
        print(f"Error parsing kwargs: {args.kwargs}")
        raise e

    logging.info(f"Running cellpose with following kwargs: {kwargs}")

    masks, flows, styles = model.eval(
        image,
        progress=True,
        channels=channels,
        **kwargs
    )

    logging.info(f"Computed masks shape:{masks.shape}")

    io.save_masks(
        images=image,
        masks=masks,
        flows=flows,
        file_names=image_name,
        png=False,
        tif=True,
        save_txt=False,
        channels=channels,
        savedir=save_dir
    )


if __name__ == '__main__':
    main()