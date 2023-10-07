import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import tifffile
import numpy as np
from cellpose_omni import io
from cellpose_omni import models
from skimage import exposure

now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--save_name', type=str, default=date_string)
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
        model_path, dim=3, nchan=1, nclasses=2, diam_mean=0, gpu=True
    )

    # Load image info
    image_path = Path(args.image_path)
    image_name = image_path.name
    image = io.imread(image_path.as_posix())
    print(f"Read image with shape: {image.shape}")

    # Normalize image
    img_min, img_max = np.percentile(image, (1, 99))
    image = exposure.rescale_intensity(image, in_range=(img_min, img_max))

    batch_size = 16
    while True:
        try:
            # Run predictions
            mask, flow = run_predictions(
                model,
                image,
                batch_size=batch_size,
                compute_masks=True,
                suppress=False,
                omni=True,
                niter=100,
                cluster=True,
                verbose=True,
                tile=True,
                bsize=114,
                channels=None,
                rescale=None,
                flow_factor=10,
                normalize=True,
                diameter=None,
                augment=True,
                mask_threshold=2,
                net_avg=False,
                min_size=4000,
                transparency=True,
                flow_threshold=0,
                hdbscan=True,
                eps=1
            )
            break

        except RuntimeError as e:
            if "out of memory" not in str(e) and "output.numel()" not in str(e):
                raise e

            # Check if batch size already 1
            if batch_size <= 1:
                raise ValueError("Out of memory error even with batch size of 1") from e

            # Reduce batch size and rerun
            print(f"Batch size of {batch_size} is too large. Halving the batch size...")
            batch_size = batch_size // 2
            torch.cuda.empty_cache()

    # Save masks
    save_dir = image_path.parent / f"{image_name}_predicted_masks"
    os.makedirs(save_dir.as_posix(), exist_ok=True)

    save_name = f"{image_name}_predicted_masks{args.save_name}.tif"
    save_path = save_dir / save_name

    print(f"Saving masks to {save_path.as_posix()}")
    tifffile.imwrite(save_path, mask)


if __name__ == '__main__':
    main()
