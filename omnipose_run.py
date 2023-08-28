import os

import tifffile
import torch
import argparse
import numpy as np

from pathlib import Path
from cellpose_omni import io
from datetime import datetime
from cellpose_omni import models
from omnipose.utils import normalize99
from cellpose_omni import io, transforms


now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--image_path', type=str, default='')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--chunks', type=int, nargs='+', default=None)
parser.add_argument('--kwargs', type=str, default=None)
parser.add_argument('--save_name', type=str, default=date_string)
parser.add_argument('--batch_num', type=str, default=None)
args = parser.parse_args()


def load_model(
        model_path: Path,
        **kwargs
) -> models.CellposeModel:
    return models.CellposeModel(
        pretrained_model="omni_test",
        **kwargs
    )

def run_predictions(
        model: models.CellposeModel,
        image: np.ndarray,
        **kwargs
):
    masks, flows, _ = model.eval(
        image,
        **kwargs
    )

    return masks, flows


def main():
    #Load model
    model_path = Path(args.model)
    model = load_model(
        model_path,
        dim=3,
        nchan=1,
        nclasses=2,
        diam_mean=0,
        gpu=True,

    )

    # Load image info
    image_path = Path(args.image_path)
    image_name = image_path.name
    image = io.imread(image_path.as_posix())

    #Run predictions
    mask, _ = run_predictions(
        model,
        image,
        omni=True,
        cluster=False,
        verbose=True,
        tile=False,
        channels=[0,0],
        rescale=None,
        flow_factor=10,
        diameter=None,
        net_avg=False,
        min_size=4000,
        transparency=True
    )

    #Save masks
    save_name = f"{image_name}_predicted_masks"
    save_path = image_path.parent / save_name
    tifffile.imwrite()


if __name__ == '__main__':
    main()






