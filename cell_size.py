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

    lower_percentile = np.percentile(counts, 25)
    upper_percentile = np.percentile(counts, 75)

    mask = (counts >= lower_percentile) & (counts <= upper_percentile)
    iqr_counts = counts[mask]

    avg_cell_size = np.cbrt(np.mean(iqr_counts))
    print(f"Average cell size in diameter (excluding outliers): {avg_cell_size}")

    cell_sizes_iqr = sorted(zip(unique[mask], iqr_counts), key=lambda x: x[1])
    for i in range(10):
        print(f"Cell value: {cell_sizes_iqr[i][0]}, Size in pixels: {cell_sizes_iqr[i][1]}")

main()

