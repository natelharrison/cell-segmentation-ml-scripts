import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Any

import omnipose
import torch
import tifffile
import numpy as np
from cellpose_omni import models, metrics
from skimage import exposure


now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='')
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--save_name', type=str, default=date_string)
args = parser.parse_args()


def load_tiff(path_str: str) -> tuple[np.ndarray, Path]:
    tif_path = Path(path_str)
    tif_array = tifffile.imread(tif_path.as_posix())

    print(f"Read tif with shape: {tif_array.shape}")
    return tif_array, tif_path


def save_tiff(
        tif_array: np.ndarray,
        tif_path: Path,
        dir_name: str,
        tiffs_processed: int = 0,

) -> None:
    save_dir = tif_path.parent / f"{dir_name}_predicted_masks"
    os.makedirs(save_dir.as_posix(), exist_ok=True)

    save_name = f"{tiffs_processed}_{tif_path.name}_predicted_masks.tif"
    save_path = save_dir / save_name

    print(f"Saving mask to {save_path.as_posix()}")
    tifffile.imwrite(save_path, tif_array)


def prediction_accuracy(masks_true: np.ndarray, masks_pred: np.ndarray):
    return metrics.average_precision([masks_true], [masks_pred], scales=[1.0])


def save_settings(
        flow_settings: dict,
        mask_settings: dict,
        tiffs_processed: int = 0,
        accuracy: Tuple = None
) -> None:
    settings = {
        "Accuracy": accuracy,
        "Flow settings": flow_settings,
        "Mask settings": mask_settings
    }

    file_name = f"{tiffs_processed}"
    if accuracy is not None:
        file_name = f"{tiffs_processed}_{accuracy[0][0]}"
    with open(f"{file_name}_settings.json", 'w') as file:
        json.dump(settings, file, indent=4)


def load_model(model_path: Path, **kwargs) -> models.CellposeModel:
    return models.CellposeModel(
        pretrained_model=model_path.as_posix(), **kwargs
    )


def run_flow_prediction(
        model: models.CellposeModel, image: np.ndarray, **kwargs
):
    masks, flows, _ = model.eval(image, **kwargs)
    return masks, flows, kwargs


def run_mask_prediction(flow, **kwargs):
    dP = flow[1]
    dist = flow[2]

    # ret is [masks_unpad, p, tr, bounds_unpad, augmented_affinity]
    ret = omnipose.core.compute_masks(dP, dist, **kwargs)
    mask = ret[0]

    return mask, kwargs


def main():
    # Load model
    model_path = Path(args.model)
    model = load_model(
        model_path, dim=3, nchan=1, nclasses=2, diam_mean=0, gpu=True
    )

    # Load image info
    image, image_path = load_tiff(args.image)

    mask_true = None
    if args.mask:
        mask_true, _ = load_tiff(args.mask)

    # Normalize image
    img_min, img_max = np.percentile(image, (1, 99))
    image = exposure.rescale_intensity(image, in_range=(img_min, img_max))

    batch_size = 8
    while True:
        try:
            # Run predictions
            mask, flow, flow_settings = run_flow_prediction(
                model,
                image,
                batch_size=batch_size,
                compute_masks=True,
                omni=True,
                niter=1,
                cluster=False,
                verbose=True,
                tile=True,
                bsize=224,
                channels=None,
                rescale=None,
                flow_factor=10,
                normalize=True,
                diameter=None,
                augment=False,
                mask_threshold=1,
                net_avg=False,
                suppress=False,
                min_size=4000,
                transparency=True,
                flow_threshold=-5
            )

            iter_list = [10, 15, 20, 25, 30, 25]
            for niter, i in enumerate(iter_list):
                mask, mask_settings = run_mask_prediction(
                    flow,
                    bd=None,
                    p=None,
                    inds=None,
                    niter=niter,
                    rescale=1,
                    resize=None,
                    mask_threshold=0,  # raise to recede boundaries
                    diam_threshold=32,
                    flow_threshold=-10,
                    interp=True,
                    cluster=False,  # speed and less under-segmentation
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
                    suppress=False,
                    eps=None,
                    hdbscan=False,
                    min_samples=6,
                    flow_factor=5,  # not needed with suppression off
                    debug=False,
                    override=False)

                # Save masks
                save_tiff(
                    mask, image_path, dir_name=args.save_name, tiffs_processed=i
                )

                if args.mask:
                    accuracy = prediction_accuracy(mask_true, mask)
                    save_settings(
                        flow_settings, mask_settings, tiffs_processed=i, accuracy=accuracy
                    )
                else:
                    save_settings(
                        flow_settings, mask_settings, tiffs_processed=i
                    )

            break

        except RuntimeError as e:
            if ("out of memory" not in str(e)
                    and "output.numel()" not in str(e)):
                raise e

            # Check if batch size already 1
            if batch_size <= 1:
                raise ValueError(
                    "Out of memory error even with batch size of 1"
                ) from e

            # Reduce batch size and rerun
            print(
                f"Batch size of {batch_size} is too large. "
                f"Halving the batch size..."
            )
            batch_size = batch_size // 2
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
