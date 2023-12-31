import gc
import os
import json
import tqdm
import logging
import argparse

import dask.array as da

from pathlib import Path
from tifffile import imwrite
from datetime import datetime
from cellpose.io import imread
from cellpose import models
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from dask_cuda import LocalCUDACluster

os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'

now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--image_path', type=str, default='')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--chunks', type=int, nargs='+', default=None)
parser.add_argument('--kwargs', type=str,
                    default='{"diameter": 30, "do_3D": true, "min_size": 2000, "augment": true, "normalize": true, "cellprob_threshold": -0.1}')
parser.add_argument('--save_name', type=str, default=date_string)
parser.add_argument('--batch_num', type=str, default=None)
parser.add_argument('--split', type=str, default=None)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


# model_type='cyto' or 'nuclei' or 'cyto2'
def load_model(
        model_path: Path,
        gpu: bool = True
) -> models.CellposeModel:
    return models.CellposeModel(gpu=gpu, pretrained_model=model_path.as_posix())


def run_predictions(model, image, channels, **kwargs):
    mask, _, _ = model.eval(image, channels=channels, batch_size=128, **kwargs)

    del image
    gc.collect()
    return mask


def tile_image(image_path: Path):
    image = imread(image_path.as_posix())
    overlap = None
    chunks = image.shape

    if args.split is not None:
        chunks = (image.shape[0]//int(args.split), image.shape[1]//int(args.split), image.shape[2]//int(args.split))
        overlap = int(image.shape[0]//10)

    if args.chunks is not None:
        chunks = args.chunks
        overlap = 128
    tiles = da.from_array(image, chunks=chunks)
    return tiles, overlap


def main():
    # Dask stuff
    with LocalCUDACluster(threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            pbar = ProgressBar()
            pbar.register()

            # Load model
            model_path = Path(args.model)
            model = load_model(model_path)

            if args.pretrained:
                model = args.pretrained

            # Load image info
            image_path = Path(args.image_path)
            image_name = image_path.name

            # File structuring
            save_name = args.save_name
            batch_num = args.batch_num
            save_dir = image_path.parent / save_name
            if batch_num is not None:
                save_dir = save_dir / model_path.stem
                image_name = f"{batch_num}_{image_name}"
            os.makedirs(save_dir, exist_ok=True)

            # Kwargs handling
            try:
                kwargs = json.loads(args.kwargs)
            except ValueError as e:
                print(f"Error parsing kwargs: {args.kwargs}")
                raise e
            logging.info(f"Running cellpose with following kwargs: {kwargs}")

            # Run predictions on tiles image using Dask
            channels = [[0, 0]]
            image_tiles, overlap = tile_image(image_path)

            if overlap is not None:
                tile_map = da.map_overlap(
                    lambda tile: run_predictions(model, tile, channels, **kwargs),
                    image_tiles,
                    depth=overlap,
                    dtype=int
                )

            else:
                tile_map = da.map_blocks(
                    lambda tile: run_predictions(model, tile, channels, **kwargs),
                    image_tiles,
                    dtype=int
                )
            with ProgressBar():
                predictions = tile_map.compute()

            imwrite(save_dir / image_name, predictions)

# def main():
#     # Load model
#     model_path = Path(args.model)
#     model = load_model(model_path)
#
#     # Load image info
#     image_path = Path(args.image_path)
#     image_name = image_path.name
#
#     # File structuring
#     save_name = args.save_name
#     batch_num = args.batch_num
#     save_dir = image_path.parent / save_name
#     if batch_num is not None:
#         save_dir = save_dir / model_path.stem
#         image_name = f"{batch_num}_{image_name}"
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Kwargs handling
#     try:
#         kwargs = json.loads(args.kwargs)
#     except ValueError as e:
#         print(f"Error parsing kwargs: {args.kwargs}")
#         raise e
#     logging.info(f"Running cellpose with following kwargs: {kwargs}")
#
#     channels = [[0, 0]]
#     image = imread(image_path.as_posix())
#
#     mask = run_predictions(model, image, channels, **kwargs)
#
#     imwrite(save_dir / image_name, mask)



if __name__ == '__main__':
    main()
