import os
import time
import logging
import argparse
import itertools

import numpy as np

from pathlib import Path
from shutil import rmtree

from skimage import exposure
from tqdm.contrib import itertools
from tifffile import imread, imwrite
from numpy.lib.stride_tricks import sliding_window_view

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--test_size', type=float, default=0)
parser.add_argument('--crop_size', type=int, nargs="+", default=(1, 64, 64))
parser.add_argument('--strides', type=int, nargs="+", default=None)
parser.add_argument('--save_name', type=str, default="processed")
parser.add_argument('--remove_label', type=int, default=0)
args = parser.parse_args()

logger = logging.getLogger(__name__)


def create_save_dir(
        dir_path: Path,
        test_files: bool = False
):
    """
    Creates a new save directory named 'cropping_output' in the provided directory path.
    If the save directory already exists, it is removed and recreated.

    :param test_files:
    :param dir_path: pathlib.Path object representing the directory in which to create the save directory
    :return: pathlib.Path object representing the created save directory
    """

    save_name = args.save_name
    dir_name = f"{save_name}_validation" if test_files else save_name
    save_dir = dir_path / dir_name
    if save_dir.exists():
        rmtree(save_dir)
    os.mkdir(save_dir)
    return save_dir


def remove_label(image):
    image[image == args.remove_label] = 0

    unique, counts = np.unique(image, return_counts=True)

    max_label_size = max(counts)
    largest_label_index = np.argmax(counts)
    largest_label = unique[largest_label_index]

    print(f"Largest label is now {largest_label} with size {max_label_size} \n")


def get_tiles(
        image_path: Path,
        save_path: Path,
        test_path: Path,
        test_size: float,
        window_size: tuple,
        strides: tuple,
        seed: int = 42):
    """
    This function takes an image and splits it into smaller tiles, saving the tiles to specified directories.

    Parameters:
    - image_path: Path to the input image.
    - save_path: Path to the directory where the tiles will be saved.
    - test_path: Path to the directory where the test tiles will be saved.
    - test_size: The proportion of tiles to be used for testing.
    - window_size: The size of the tiles.
    - strides: The strides to use when splitting the image into tiles.
    - split: Whether to split the image into six equal parts before tiling.
    - seed: Seed for the random number generator. This is used when selecting the test tiles.
    """

    if image_path.suffix == '.tif':
        image = imread(image_path.as_posix())
    else:
        logger.error(f"Unknown file format: {image_path.name}")
        return

    if strides is None:
        strides = window_size

    if args.remove_label and 'mask' in image_path.stem:
        remove_label(image)

    print(f"Tiles will be saved to {save_path}")
    # Create crops along XY, ZY, and ZX axes

    # Normalize image
    img_min, img_max = np.percentile(image, (1, 99))
    image = exposure.rescale_intensity(image, in_range=(img_min, img_max))

    dims = 1 if args.crop_size[0] != 1 else 3

    for axis in range(dims):
        # Rotate the image array along the current axis
        if dims == 1:
            rotated_image = image
        else:
            rotated_image = np.rot90(image, axes=(axis, (axis + 1) % 3))

        print(rotated_image.shape)

        windows = sliding_window_view(
            rotated_image, window_shape=window_size)[::strides[0], ::strides[1], ::strides[2]]
        z_tiles, n_rows, n_cols = windows.shape[:3]
        windows = np.reshape(windows, (-1, *window_size))

        test_count = int(len(windows) * test_size)

        np.random.seed(seed)
        test_indices = np.random.choice(len(windows), test_count, replace=False)

        # Define plane names
        plane_names = ['XY', 'ZY', 'ZX']

        tiles_skipped = 0
        for i, (z, y, x) in enumerate(itertools.product(
                range(z_tiles), range(n_rows), range(n_cols),
                desc=f"Locating tiles in {plane_names[axis]}: {[windows.shape[0]]}",
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=' tile',
        )):
            # Include the plane name in the tile name
            tile_name = f"{plane_names[axis]}_z{z}-y{y}-x{x}_{image_path.stem}"
            tile = windows[i]
            if list(tile.shape) != window_size:
                tiles_skipped += 1
                continue

            if i in test_indices:
                imwrite(test_path / f"{tile_name}.tif", tile)
            else:
                imwrite(save_path / f"{tile_name}.tif", tile)
        print(f"{tiles_skipped} tiles discarded in axis {axis}")

def main():
    # Start timer
    start_time = time.time()

    test_size = args.test_size
    window_size = args.crop_size
    strides = args.strides

    # Create save directory
    dir = Path(args.dir)

    test_dir = Path()
    save_dir = create_save_dir(dir)
    if test_size:
        test_dir = create_save_dir(dir, True)

    # Get Images
    image_files = list(dir.glob('*tif'))
    total_images = len(image_files)

    print(f'Images Read: {total_images}')

    seed = np.random.randint(1000)

    for file in image_files:
        img_path = Path(file)
        get_tiles(img_path,
                  save_dir,
                  test_dir,
                  test_size,
                  window_size,
                  strides,
                  seed)

    # End timer and print execution time
    print(f'--- {time.time() - start_time} seconds ---')


main()
