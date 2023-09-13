import argparse
import torch
import omnipose
import tifffile
import numpy as np

from pathlib import Path
from datetime import datetime
from cellpose_omni import io, core
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
    torch.cuda.empty_cache()

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
    mask, flow = run_predictions(
        model,
        image,
        compute_masks=True,
        omni=True,
        batch_size=6,
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
        min_size=4000,
        transparency=True,
        flow_threshold=-5
    )
    masks = [mask]
    flows = [flow]

    k = 0
    dP = flows[k][1]
    dist = flows[k][2]
    # ret is [masks_unpad, p, tr, bounds_unpad, augmented_affinity]
    ret = omnipose.core.compute_masks(
        dP,
        dist,
        bd=None,
        p=None,
        inds=None,
        niter=150,
        rescale=1.0,
        resize=None,
        mask_threshold=0,  # raise this higher to recede boundaries
        diam_threshold=25,
        flow_threshold=0,
        interp=True,
        cluster=False,  # speed and less undersegmentation
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
        flow_factor=1,  # not needed with supression off and niter set manually
        debug=False,
        override=False)

    masks[k] = ret[0]

    # Save masks
    save_name = f"{image_name}_predicted_masks.tif"
    save_path = image_path.parent / save_name
    tifffile.imwrite(save_path, mask)


if __name__ == '__main__':
    main()
