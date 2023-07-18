import itertools
import logging
import os
import re
import time
import argparse

import numpy as np

from pathlib import Path
from shutil import rmtree
from tifffile import imread, imwrite
from tqdm.contrib import itertools
from numpy.lib.stride_tricks import sliding_window_view

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--test_size', type=float, default=0)
parser.add_argument('--crop_size', type=int, nargs="+", default=(1, 64, 64))
parser.add_argument('--strides', type=int, nargs="+", default=None)
parser.add_argument('--save_name', type=str, default="processed")
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


def get_cleaned_file_name(file):
    """
    Extract the cleaned file name from the given file using a regular expression pattern.

    :param file: pathlib.Path object representing the file
    :return: string, the cleaned file name
    """
    pattern = 'Scan_Iter_\d{4}_CamA_ch0_CAM1_stack\d{4}_\d{3}nm_\d{7}msec_\d{10}msecAbs'
    return re.match(pattern, file.stem)[0]


def get_tiles(
        image_path: Path,
        save_path: Path,
        test_path: Path,
        test_size: int,
        window_size: tuple,
        strides: tuple,
        seed: int = 42):
    """
    This function takes an image and splits it into smaller tiles, saving the tiles to specified directories.

    Parameters:
    - image_path: Path to the input image.
    - save_path: Path to the directory where the tiles will be saved.
    - test_path: Path to the directory where the test tiles will be saved.
    - test_size: The proportion of tiles to be used for testing. This should be a float between 0 and 1.
    - window_size: The size of the tiles. This should be a tuple of three integers.
    - strides: The strides to use when splitting the image into tiles. This should be a tuple of three integers.
    - split: Whether to split the image into six equal parts before tiling.
    - seed: Seed for the random number generator. This is used when selecting the test tiles.

    The function reads the image from the specified path, splits it into tiles of the specified size, and saves the tiles to the specified directories. The tiles are saved in TIFF format. The function also supports splitting the image into six equal parts before tiling, which can be useful for large images.

    Note: This function assumes that the input image is a 3D image stored in a TIFF file.
    """

    if image_path.suffix == '.tif':
        image = imread(image_path.as_posix())
    else:
        logger.error(f"Unknown file format: {image_path.name}")
        return

    if strides is None:
        strides = window_size

    windows = sliding_window_view(image, window_shape=window_size)[::strides[0], ::strides[1], ::strides[2]]
    z_tiles, n_rows, n_cols = windows.shape[:3]
    windows = np.reshape(windows, (-1, *window_size))

    test_count = int(len(windows) * test_size)

    np.random.seed(seed)
    test_indices = np.random.choice(len(windows), test_count, replace=False)

    for i, (z, y, x) in enumerate(itertools.product(
            range(z_tiles), range(n_rows), range(n_cols),
            desc=f"Locating tiles: {[windows.shape[0]]}",
            bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
            unit=' tile',
    )):
        tile = f"z{z}-y{y}-x{x}_{image_path.stem}"
        if i in test_indices:
            imwrite(test_path / f"{tile}.tif", windows[i])
        else:
            imwrite(save_path / f"{tile}.tif", windows[i])


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
