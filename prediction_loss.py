import os
import torch
import argparse
from pathlib import Path
from cellpose import models
from cellpose.io import imread

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

    model = models.CellposeModel()

    labels = imread(labels_path.as_posix())
    labels_tensor = torch.from_numpy(labels)[None, ...]

    files = list_files(dir_path)
    for file in files:
        prediction = imread(file.as_posix())
        prediction_tensor = torch.from_numpy(prediction)[None, ...]

        loss = model.loss_fn(labels_tensor, prediction_tensor)
        print(f"File: {file.as_posix()}, Loss: {loss.item()}")

if __name__ == '__main__':
    main()