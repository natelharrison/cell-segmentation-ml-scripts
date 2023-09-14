import time
import torch
import argparse
import omnipose
import tifffile
import threading
import numpy as np
import matplotlib.pyplot as plt

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

gpu_memory_logs = []


def monitor_gpu(interval=2):
    num_gpus = torch.cuda.device_count()
    while True:
        mem_info = [torch.cuda.memory_allocated(device=i) / 1024 ** 3 for i in range(num_gpus)]
        gpu_memory_logs.append(mem_info)
        print(f"GPU Memory Used: {mem_info} GiB")
        time.sleep(interval)


def plot_memory_usage():
    num_gpus = torch.cuda.device_count()
    plt.figure(figsize=(10, 6))

    for i in range(num_gpus):
        plt.plot([log[i] for log in gpu_memory_logs], label=f'GPU {i}')

    plt.title('GPU Memory Usage Over Time')
    plt.xlabel('Time (every 2 seconds)')
    plt.ylabel('Memory Used (MiB)')
    plt.legend()
    plt.grid(True)
    plt.show()


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
    mask, flow = run_predictions(
        model,
        image,
        compute_masks=True,
        omni=True,
        batch_size=4,
        niter=170,
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
        mask_threshold=0,
        net_avg=False,
        min_size=4000,
        transparency=True,
        flow_threshold=0
    )

    # Save masks
    save_name = f"{image_name}_predicted_masks.tif"
    save_path = image_path.parent / save_name
    tifffile.imwrite(save_path, mask)


if __name__ == '__main__':
    t = threading.Thread(target=monitor_gpu)
    t.start()

    main()

    t.join()
    plot_memory_usage()
