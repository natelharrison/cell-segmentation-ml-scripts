import argparse

import numpy as np

from tifffile import imread
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str)
args = parser.parse_args()


def main():
    file_path = Path(args.image_path)

    image = imread(file_path.as_posix())
    image = image.flatten()

    unique, counts = np.unique(image, return_counts=True)

    # Pair each unique pixel value with its count and sort by count
    cell_sizes = sorted(zip(unique, counts), key=lambda x: x[1])

    # Print the 10 smallest cell sizes
    for i in range(10):
        print(f"Cell value: {cell_sizes[i][0]}, Size in pixels: {cell_sizes[i][1]}")

main()
