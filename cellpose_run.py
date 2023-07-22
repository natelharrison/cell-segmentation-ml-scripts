import ast
import os
import argparse
import json
from pathlib import Path
import logging
from cellpose import models, io
from cellpose.io import imread


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--image_path', type=str, default='')
parser.add_argument('--model', type=str, default='cyto2')
parser.add_argument('--kwargs', type=str, default='{}')
parser.add_argument('--save_name', type=str, default=None)
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

    file_path = Path(args.image_path)
    file_name = file_path.stem
    file = file_path.as_posix()

    save_dir = file_path.parent / f"{model_path.stem}_predictions"
    if args.save_name:
        save_dir = save_dir / args.save_name
    os.makedirs(save_dir, exist_ok=True)
    save_dir = save_dir.as_posix()

    channels = [[0,0]]
    image = imread(file)

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
        diameter=30
    )
    io.save_masks(
        images=image,
        masks=masks,
        flows=flows,
        file_names=file_name,
        png=False,
        tif=True,
        save_txt=False,
        channels=channels,
        savedir=save_dir
    )


if __name__ == '__main__':
    main()