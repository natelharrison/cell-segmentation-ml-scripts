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


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
args = parser.parse_args()

def create_save_dir(dir_path: Path):
    ...
def filter_labels(image: np.ndarray):
    ...
def tile_stitching(tiles: list):
    ...
def postprocessing(image_files: list):
    
    return

def main():

    dir = Path(args.dir)
    image_files = list(dir.glob('*.tif'))
    print(f"Images found:{len(image_files)}")

    postprocessing(image_files)

main()









