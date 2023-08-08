import argparse

import numpy as np

from tifffile import imread
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str)
parser.add_argument('--remove_label', type=int)
args = parser.parse_args()


def main():
    file_path = Path(args.image_path)

    image = imread(file_path.as_posix())
    image = image.flatten()

    if args.remove_label:
        image[image == args.remove_label] = 0

    unique, counts = np.unique(image, return_counts=True)

    lower_percentile = np.percentile(counts, 25)
    upper_percentile = np.percentile(counts, 75)

    mask = (counts >= lower_percentile) & (counts <= upper_percentile)
    iqr_counts = counts[mask]

    max_label_size = max(counts)
    largest_label_index = np.argmax(counts)
    largest_label = unique[largest_label_index]
    print(f"Largest label is {largest_label} with size {max_label_size} \n")

    avg_cell_size = np.cbrt(np.mean(iqr_counts))
    print(f"Average cell size in diameter (excluding outliers): {avg_cell_size}")

    cell_sizes_iqr = sorted(zip(unique[mask], iqr_counts), key=lambda x: x[1])
    for i in range(10):
        print(f"Cell value: {cell_sizes_iqr[i][0]}, Size in pixels: {cell_sizes_iqr[i][1]}")

    cell_sizes = sorted(zip(unique, counts), key=lambda x: x[1])
    print("\nMinimum cell sizes including outliers:")
    for i in range(10):
        print(f"Cell value: {cell_sizes[i][0]}, Size in pixels: {cell_sizes[i][1]}")

main()


