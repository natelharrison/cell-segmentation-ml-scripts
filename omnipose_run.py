import argparse
import tifffile
import numpy as np

from pathlib import Path
from datetime import datetime
from cellpose_omni import io
from cellpose_omni import models

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


def load_model(model_path: Path, **kwargs) -> models.CellposeModel:
    return models.CellposeModel(
        pretrained_model=model_path.as_posix(), **kwargs
    )


def run_predictions(
        model: models.CellposeModel, image: np.ndarray, **kwargs
):
    masks, flows, _ = model.eval(image, **kwargs)
    return masks, flows


def main():
    # Load model
    model_path = Path(args.model)
    model = load_model(
        model_path, dim=3, nchan=1, nclasses=2, diam_mean=0, gpu=True,
    )

    # Load image info
    image_path = Path(args.image_path)
    image_name = image_path.name
    image = io.imread(image_path.as_posix())
    print(f"Read image with shape: {image.shape}")

    # Run predictions
    mask, _ = run_predictions(
        model,
        image,

        compute_masks=True,
        omni=True,
        augment=True,
        suppress=False,
        verbose=True,
        tile=True,
        niter=170,
        batch_size=8,
        flow_factor=10,
        mask_threshold=0,
        flow_threshold=0,
        rescale=None,
        channels=None,
        normalize=True,
        diameter=None,
        min_size=4000,
        diam_threshold=30,
        cluster=False,
        net_avg=False,
        transparency=True,
    )

    # Save masks
    save_name = f"{image_name}_predicted_masks.tif"
    save_path = image_path.parent / save_name
    tifffile.imwrite(save_path, mask)


if __name__ == '__main__':
    main()
