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

def main():
    dir_path = Path(args.dir)
    labels_path = Path(args.labels)

    labels = imread(labels_path.as_posix())
    labels = labels.astype('float32')[None, ...]

    files = list_files(dir_path)
    model = CellposeModel(gpu=False, model_type='cyto')

    for file in files:
        prediction = imread(file.as_posix())
        prediction = prediction.astype('float32')[None, ...]

        loss = model.loss_fn(labels, prediction)
        print(f"File: {file.as_posix()}, Loss: {loss.item()}")


if __name__ == '__main__':
    main()
