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
    num_classes = np.max(true) + 1  # Assumes labels are in range [0, num_classes)
    dice = 0
    for i in range(num_classes):
        true_binary = (true == i)
        preds_binary = (preds == i)
        intersection = np.sum(true_binary * preds_binary)
        dice += (2. * intersection) / (np.sum(true_binary) + np.sum(preds_binary))
    return 1 - dice / num_classes  # Average over all classes



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
