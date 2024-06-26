import argparse
import os
from datetime import datetime
from pathlib import Path

import omnipose
import torch
import tifffile
import numpy as np
from cellpose_omni import models, metrics
from skimage import exposure
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer, Categorical

now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to an image file or directory containing image files.")
parser.add_argument('--reference_image', type=str, default=None)
parser.add_argument('--mask', type=str, default=None)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--save_name', type=str, default=date_string)
parser.add_argument('--mask_settings', type=float, default=None)
parser.add_argument('--flows', type=str, default=None)
parser.add_argument('--save_flows', action='store_true', default=False)
args = parser.parse_args()


def prediction_accuracy(
        masks_predicted: np.ndarray = None,
        masks_true: np.ndarray = None,
        dP: np.ndarray = None,
        model: models.CellposeModel = None
):
    # ap, _, _, _ = metrics.average_precision(
    #     [masks_true], [masks_predicted]
    # )
    # print(ap[0])
    # return 1 - ap[0][0]  # least strict threshold

    metrics.flow_error(masks_predicted, dP, use_gpu=False, device=None)
    try:
        flow_errors, _ = metrics.flow_error(masks_predicted, dP)
        return np.mean(flow_errors)
    except TypeError as e:
        if "NoneType" not in str(e):
            raise e
        print("Ran into Type Error: cannot unpack non-iterable NoneType object")
        return 999


def prediction_optimization(
        model: models.CellposeModel,
        flow: np.ndarray,
        mask_true: np.ndarray,
) -> None:
    search_space = [
        Integer(64, 160, name='niter'),
        Real(-5, 5, name='mask_threshold'),
        Integer(0, 32, name='diam_threshold'),
        Real(-4, 0, name='flow_threshold'),
        Integer(0, 32000, name='min_size'),
    ]

    @use_named_args(search_space)
    def objective(**kwargs):
        print(f"Testing with values {kwargs}")
        torch.cuda.empty_cache()
        mask, mask_settings = run_mask_prediction(
            flow,
            niter=kwargs['niter'],
            mask_threshold=kwargs['mask_threshold'],  # raise to recede boundaries
            diam_threshold=kwargs['diam_threshold'],
            flow_threshold=kwargs['flow_threshold'],
            min_size=kwargs['min_size'],
            device=model.device
        )

        dP = flow[1]
        score = prediction_accuracy(
            masks_predicted=mask,
            dP=dP,
            model=model
        )
        print(score)
        return score

    # Run Bayesian optimization
    res_gp = gp_minimize(objective, search_space, n_calls=32, random_state=0)

    # Results
    print("Best parameters: {}".format(res_gp.x))
    print("Best score: {}".format(res_gp.fun))


def load_images(input_path
) -> None:
    path = Path(input_path)
    if path.is_dir():
        return [p for p in path.glob('*.tif')]
    elif path.is_file():
        return [path]
    else:
        raise ValueError(f"No valid image or directory found at {input_path}")


def load_tiff(path_str: str) -> tuple[np.ndarray, Path]:
    tif_path = Path(path_str)
    tif_array = tifffile.imread(tif_path.as_posix())

    print(f"Read tif with shape: {tif_array.shape}")
    return tif_array, tif_path


def save_tiff(
        tif_array: np.ndarray,
        tif_path: Path,
        dir_name: str,
        data_type: str = 'masks'

) -> Path:
    save_dir = tif_path.parent / f"{dir_name}_predicted_{data_type}"
    os.makedirs(save_dir.as_posix(), exist_ok=True)

    save_name = f"{tif_path.name}_predicted_{data_type}.tif"
    save_path = save_dir / save_name

    print(f"Saving {data_type} to {save_path.as_posix()}")
    tifffile.imwrite(save_path, tif_array)
    return save_dir


def load_model(model_path: Path, **kwargs) -> models.CellposeModel:
    return models.CellposeModel(
        pretrained_model=model_path.as_posix(), **kwargs
    )


def run_flow_prediction(
        model: models.CellposeModel,
        image: np.ndarray,
        ref_image: np.ndarray = None,
        **kwargs,
):
    # If reference image (training data) is provided match histograms
    if ref_image is not None:
        image = exposure.match_histograms(image, ref_image)

    # Normalize image
    img_min, img_max = np.percentile(image, (1, 99))
    image = exposure.rescale_intensity(image, in_range=(img_min, img_max))

    batch_size = 16
    while True:
        try:
            _, flows, _ = model.eval(
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
                flow_threshold=-5,
                **kwargs)

            return _, flows, _

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


def run_mask_prediction(flow, **kwargs):
    dP = flow[1]
    dist = flow[2]

    # ret is [masks_unpad, p, tr, bounds_unpad, augmented_affinity]
    ret = omnipose.core.compute_masks(
        dP,
        dist,
        bd=None,
        p=None,
        rescale=1.0,
        resize=None,
        interp=True,
        cluster=False,  # speed and less under-segmentation
        boundary_seg=False,
        affinity_seg=False,
        do_3D=False,
        max_size=None,
        hole_size=None,
        omni=True,
        calc_trace=False,
        verbose=True,
        use_gpu=True,
        nclasses=2,
        dim=3,
        suppress=False,
        eps=None,
        hdbscan=False,
        flow_factor=5,  # not needed with suppression off
        debug=False,
        override=False,
        **kwargs)
    mask = ret[0]

    return mask, kwargs


def main():
    model_path = Path(args.model)
    model = load_model(model_path, dim=3, nchan=1, nclasses=2, diam_mean=0, gpu=True)

    image_paths = load_images(args.input)

    for image_path in image_paths:
        image, _ = load_tiff(image_path)

        ref_image = None
        if args.reference_image is not None:
            ref_image, _ = load_tiff(args.reference_image)

        mask_true = None
        if args.mask is not None:
            mask_true, _ = load_tiff(args.mask)

        flow = None
        if args.flows is not None:
            flow, _ = load_tiff(args.flows)
        else:
            # Run flow predictions
            _, flow, _ = run_flow_prediction(model, image, ref_image=ref_image)

        if args.save_flows:
            multi_channel_flow_data = np.stack(flow, axis=0)
            _ = save_tiff(multi_channel_flow_data, image_path, args.save_name)
            continue

        if mask_true is not None:
            prediction_optimization(model, flow, mask_true)
            continue

        mask, mask_settings = run_mask_prediction(
            flow,
            device=model.device,
            niter=30,  # default settings
            mask_threshold=2.0,
            diam_threshold=32,
            flow_threshold=0.0,
            min_size=11000,
        )

        _ = save_tiff(mask, image_path, args.save_name)


if __name__ == '__main__':
    main()
