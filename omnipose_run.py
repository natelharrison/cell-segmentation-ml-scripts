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
stop_event = threading.Event()


def monitor_gpu(interval=15):
    num_gpus = torch.cuda.device_count()
    while not stop_event.is_set():
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
        model_path, dim=3, nchan=1, nclasses=2, diam_mean=0, gpu=True
    )

    # Load image info
    image_path = Path(args.image_path)
    image_name = image_path.name
    image = io.imread(image_path.as_posix())
    print(f"Read image with shape: {image.shape}")

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
            break

        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise e

            if batch_size <= 1:  # Check if batch size is already 1
                raise ValueError("Out of memory error even with batch size of 1") from e

            print(f"Batch size of {batch_size} is too large. Halving the batch size...")
            batch_size = batch_size // 2
            torch.cuda.empty_cache()

    # Save masks
    save_name = f"{image_name}_predicted_masks.tif"
    save_dir = image_path.parent / f"{image_name}_predicted_masks"
    save_path = save_dir / save_name / date_string
    print(f"Saving masks to {save_path.as_posix()}")
    tifffile.imwrite(save_path, mask)


if __name__ == '__main__':
    t = threading.Thread(target=monitor_gpu)
    t.start()

    main()

    stop_event.set()
    t.join()
    plot_memory_usage()
