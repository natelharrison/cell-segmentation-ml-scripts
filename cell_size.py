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

    _, counts = np.unique(image, return_counts=True)

    image = np.sort(image)

    for i in range(10):
        print(image[i], "\n")

main()
