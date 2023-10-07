import os
import argparse
from pathlib import Path
from datetime import datetime

import omnipose
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
    # img_min, img_max = np.percentile(image, (1, 99))
    # image = exposure.rescale_intensity(image, in_range=(img_min, img_max))

    batch_size = 16
    while True:
        try:
            # Run predictions
            _, flow = run_predictions(
                model,
                image,
                batch_size=batch_size,
                compute_masks=True,
                omni=True,
                niter=1,
                cluster=False,
                verbose=True,
                tile=True,
                bsize=112,
                channels=None,
                rescale=None,
                flow_factor=10,
                normalize=True,
                diameter=None,
                augment=False,
                mask_threshold=1,
                net_avg=False,
                min_size=4000,
                transparency=True,
                flow_threshold=-5
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

    dP = flow[1]
    dist = flow[2]
    # ret is [masks_unpad, p, tr, bounds_unpad, augmented_affinity]

    ret = omnipose.core.compute_masks(
        dP,
        dist,
        bd=None,
        p=None,
        inds=None,
        niter=100,
        rescale=2.0,
        resize=None,
        mask_threshold=2,  # raise this higher to recede boundaries
        diam_threshold=32,
        flow_threshold=0,
        interp=True,
        cluster=True,  # speed and less undersegmentation
        boundary_seg=False,
        affinity_seg=False,
        do_3D=False,
        min_size=4000,
        max_size=None,
        hole_size=None,
        omni=True,
        calc_trace=False,
        verbose=True,
        use_gpu=True,
        device=model.device,
        nclasses=2,
        dim=3,
        suppress=False,  # this option opened up now
        eps=None,
        hdbscan=False,
        flow_factor=5,  # not needed with supression off and niter set manually
        debug=False,
        override=False)

    mask = ret[0]

    # Save masks
    save_dir = image_path.parent / f"{image_name}_predicted_masks"
    os.makedirs(save_dir.as_posix(), exist_ok=True)

    save_name = f"{image_name}_predicted_masks{args.save_name}.tif"
    save_path = save_dir / save_name

    print(f"Saving masks to {save_path.as_posix()}")
    tifffile.imwrite(save_path, mask)


if __name__ == '__main__':
    main()
