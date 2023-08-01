import argparse
import numpy as np
from cellpose import utils
from pathlib import Path
from cellpose.io import imread
from cellpose.models import CellposeModel

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--labels', type=str, default='')
args = parser.parse_args()

def list_files(directory):
    files = []
    base_dir = Path(directory)
    for file in base_dir.rglob('*'):
        if file.is_file() and file.parent != base_dir:
            files.append(file)
    return files


def dice_loss(true, preds):
    num_classes = int(np.max(true)) + 1
    true_one_hot = np.eye(num_classes)[true.astype(int)]
    preds_one_hot = np.eye(num_classes)[preds.astype(int)]
    intersection = np.sum(true_one_hot * preds_one_hot, axis=(-1, -2, -3))
    dice = (2. * intersection) / (np.sum(true_one_hot, axis=(-1, -2, -3)) + np.sum(preds_one_hot, axis=(-1, -2, -3)))
    return 1 - np.mean(dice)


def main():
    dir_path = Path(args.dir)
    labels_path = Path(args.labels)

    labels = imread(labels_path.as_posix())

    files = list_files(dir_path)

    for file in files:
        prediction = imread(file.as_posix())

        loss = dice_loss(labels, prediction)
        print(f"File: {file.as_posix()}, Loss: {loss}")



if __name__ == '__main__':
    main()
