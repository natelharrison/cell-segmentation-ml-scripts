import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import omnipose
import torch
import tifffile
import numpy as np
from cellpose_omni import models, metrics
from skimage import exposure

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer

now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None)
parser.add_argument('--mask', type=str, default=None)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--save_name', type=str, default=date_string)
args = parser.parse_args()


def prediction_accuracy(masks_true: np.ndarray, masks_predicted: np.ndarray):
    ap, _, _, _ = metrics.average_precision(
        [masks_true], [masks_predicted]
    )
    return np.mean(ap)


def prediction_optimization(
        model: models.CellposeModel,
        flow: np.ndarray,
        mask_true: np.ndarray,
) -> None:
    search_space = [
        Integer(10, 30, name='niter'),
        Real(-1, 1, name='mask_threshold'),
        Integer(8, 32, name='diam_threshold'),
        Real(-1, 0, name='flow_threshold'),
        Integer(4000, 32000, name='min_size')
    ]

    @use_named_args(search_space)
    def objective(**kwargs):
        mask, mask_settings = run_mask_prediction(
            flow,
            bd=None,
            p=None,
            niter=kwargs['niter'],
            rescale=1,
            resize=None,
            mask_threshold=kwargs['mask_threshold'],  # raise to recede boundaries
            diam_threshold=kwargs['diam_threshold'],
            flow_threshold=kwargs['flow_threshold'],
            interp=True,
            cluster=False,  # speed and less under-segmentation
            boundary_seg=False,
            affinity_seg=False,
            do_3D=False,
            min_size=kwargs['min_size'],
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
            flow_factor=5,  # not needed with suppression off
            debug=False,
            override=False)

        score = prediction_accuracy(mask_true, mask)
        print(**kwargs)
        print(score)
        return score

    # Run Bayesian optimization
    res_gp = gp_minimize(objective, search_space, n_calls=10, random_state=0)

    # Results
    print("Best parameters: {}".format(res_gp.x))
    print("Best score: {}".format(res_gp.fun))


def load_tiff(path_str: str) -> tuple[np.ndarray, Path]:
    tif_path = Path(path_str)
    tif_array = tifffile.imread(tif_path.as_posix())

    print(f"Read tif with shape: {tif_array.shape}")
    return tif_array, tif_path


def save_tiff(
        tif_array: np.ndarray,
        tif_path: Path,
        dir_name: str,

) -> Path:
    save_dir = tif_path.parent / f"{dir_name}_predicted_masks"
    os.makedirs(save_dir.as_posix(), exist_ok=True)

    save_name = f"{tif_path.name}_predicted_masks.tif"
    save_path = save_dir / save_name

    print(f"Saving mask to {save_path.as_posix()}")
    tifffile.imwrite(save_path, tif_array)
    return save_dir


def load_model(model_path: Path, **kwargs) -> models.CellposeModel:
    return models.CellposeModel(
        pretrained_model=model_path.as_posix(), **kwargs
    )


def run_flow_prediction(
        model: models.CellposeModel, image: np.ndarray, **kwargs
):
    # Normalize image
    img_min, img_max = np.percentile(image, (1, 99))
    image = exposure.rescale_intensity(image, in_range=(img_min, img_max))

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
    print("This si the most recent code...")
    # Load model
    model_path = Path(args.model)
    model = load_model(
        model_path, dim=3, nchan=1, nclasses=2, diam_mean=0, gpu=True
    )

    # Load image and ground truth mask if provided
    image, image_path = load_tiff(args.image)
    mask_true = None
    if args.mask is not None:
        mask_true, _ = load_tiff(args.mask)

    # Run  flow predictions
    batch_size = 8
    while True:
        try:
            _, flow, flow_settings = run_flow_prediction(
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
                augment=True,
                mask_threshold=1,
                net_avg=False,
                suppress=False,
                min_size=4000,
                transparency=True,
                flow_threshold=-5
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
                f"Batch size of {batch_size} is too large.  "
            )
            batch_size = batch_size // 2
            torch.cuda.empty_cache()

    # If ground truth provided, optimize parameters and print results
    if mask_true is not None:
        prediction_optimization(model, flow, mask_true)

    else:
        # Run mask predictions
        niter = 20
        mask_threshold = 0
        diam_threshold = 0
        flow_threshold = 0
        min_size = 4000

        mask, mask_settings = run_mask_prediction(
            flow,
            bd=None,
            p=None,
            niter=niter,
            rescale=1,
            resize=None,
            mask_threshold=mask_threshold,  # raise to recede boundaries
            diam_threshold=diam_threshold,
            flow_threshold=flow_threshold,
            interp=True,
            cluster=False,  # speed and less under-segmentation
            boundary_seg=False,
            affinity_seg=False,
            do_3D=False,
            min_size=min_size,
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
            flow_factor=5,  # not needed with suppression off
            debug=False,
            override=False)

        # Save masks
        _ = save_tiff(mask, image_path, dir_name=args.save_name)


if __name__ == '__main__':
    main()
