import gc
import os
import argparse
import sys

from typing import Tuple, Union, Optional

import ray
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from numpy import ndarray
from pathlib import Path

from skimage import exposure
from tifffile import imread, imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='Path to the input image')
parser.add_argument('--mask_path', type=str, help='Path to the input mask')
parser.add_argument('--num_chunks', type=int, default=os.cpu_count())
parser.add_argument('--object_store_memory', type=int, default=None)
parser.add_argument('--background', type=int, help='Remove background if labeled')
parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization')
args = parser.parse_args()

if args.object_store_memory is None:
    ray.init(address='auto')
else:
    object_store_memory = args.object_store_memory * 10 ** 9
    ray.init(address='auto',
             object_store_memory=object_store_memory,
             num_cpus=args.num_chunks)


def get_label_slice(mask: np.ndarray) -> ndarray[int]:
    """
    Get the index of the slice with the most prominent label in a 3D binary mask.
    :param mask: np.ndarray: The 3D binary mask.
    :return ndarray[int]: The index of the prominent slice.
    """
    slice_sums = mask.sum(axis=(1, 2))
    return np.argmax(slice_sums)


def visualize_3d_slice(
        image: np.ndarray,
        mask: np.ndarray,
        gradient_magnitude: np.ndarray,
        refined_mask: np.ndarray,
):
    """
    Visualize slices of input image, mask, gradient magnitude, and refined mask.

    :param gradient_magnitude: The initial mask.
    :param refined_mask: The gradient magnitude.
    :param image: The input image.
    :param mask: The refined mask.
    """
    slice_index = get_label_slice(mask)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Display the input image slice
    axes[0].imshow(image[slice_index], cmap='gray')
    axes[0].set_title('Input Image Slice')
    axes[0].axis('off')

    # Display the initial mask slice
    axes[1].imshow(gradient_magnitude[slice_index], cmap='jet')
    axes[1].set_title('Gradient Magnitude Slice')
    axes[1].axis('off')

    # Display the gradient magnitude slice
    axes[2].imshow(refined_mask[slice_index], cmap='jet')
    axes[2].set_title('Refined Mask Slice')
    axes[2].axis('off')

    # Display the refined mask slice
    axes[3].imshow(mask[slice_index], cmap='gray')
    axes[3].set_title('Initial Mask Slice')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


def update_mask(refined_mask, results):
    for binary_mask, label in results:
        if binary_mask is not None:
            refined_mask[binary_mask] = label
    return refined_mask


def get_bounding_box(
        binary_mask: np.ndarray, label: int, buffer: int = 16
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Returns the bounding box for a given label
    :param label:
    :param binary_mask:
    :param buffer:
    :return bounding_coords:
    """
    x, y, z = np.where(binary_mask)

    if x.size == 0 or y.size == 0 or z.size == 0:
        print(f"Label {label} has no positive pixels in the binary mask.")
        return None

    x_min, x_max = np.min(x) - buffer, np.max(x) + buffer + 1
    y_min, y_max = np.min(y) - buffer, np.max(y) + buffer + 1
    z_min, z_max = np.min(z) - buffer, np.max(z) + buffer + 1

    # Make sure coordinated within image dimensions
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    z_min = max(0, z_min)

    x_max = min(binary_mask.shape[0], x_max)
    y_max = min(binary_mask.shape[1], y_max)
    z_max = min(binary_mask.shape[2], z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max


@ray.remote
def process_label(
        mask: np.ndarray,
        image: np.ndarray,
        label: int
) -> Tuple[Optional[np.ndarray[np.bool_]], int]:
    """
    Process a label within a Pool of processes.
    :param mask:
    :param image:
    :param label:
    :return active_contour():
    """
    binary_mask = (mask == label).astype(np.bool_)

    x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(binary_mask, label)
    cropped_image = image[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_mask = binary_mask[x_min:x_max, y_min:y_max, z_min:z_max]

    if np.sum(binary_mask) <= 512 or cropped_mask is None:
        print("Label contains fewer than 512 labeled pixels - skipping")
        sys.stdout.flush()
        return None, label

    refined_cropped_label = active_contour(cropped_image, cropped_mask)

    binary_mask[
        x_min: x_max, y_min: y_max, z_min: z_max
    ] = refined_cropped_label

    del refined_cropped_label, cropped_mask, cropped_image
    gc.collect()

    print(f"Processed label {label}")
    sys.stdout.flush()

    return binary_mask.astype(np.bool_), label


def active_contour(
        image: np.ndarray,
        binary_mask: np.ndarray
) -> np.ndarray[np.bool_]:
    """
    Perform active contour segmentation on an input image.
    :param image:
    :param binary_mask:
    :return refined_mask as binary array:
    """
    itk_image = sitk.Cast(sitk.GetImageFromArray(image), sitk.sitkFloat32)
    itk_binary_mask = sitk.GetImageFromArray(binary_mask.astype(np.uint8))

    # Convert binary mask to a signed distance map
    itk_mask = sitk.SignedMaurerDistanceMap(
        itk_binary_mask,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True
    )

    gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussian(
        itk_image, sigma=4
    )

    # Geodesic active contour filter initialization
    img_filter = sitk.GeodesicActiveContourLevelSetImageFilter()
    img_filter.SetPropagationScaling(-2)
    img_filter.SetCurvatureScaling(7.5)
    img_filter.SetAdvectionScaling(20.0)
    img_filter.SetMaximumRMSError(0.005)
    img_filter.SetNumberOfIterations(500)

    refined_mask = img_filter.Execute(itk_mask, gradient_magnitude)

    if args.visualize:
        visualize_3d_slice(
            image,
            binary_mask,
            sitk.GetArrayFromImage(gradient_magnitude),
            sitk.GetArrayFromImage(refined_mask).astype(np.float16)
        )

    # Convert the refined mask into binary
    return (sitk.GetArrayFromImage(refined_mask) > -1).astype(np.bool_)


def main():
    image_path: Union[str, Path] = Path(args.image_path)
    mask_path: Union[str, Path] = Path(args.mask_path)

    image = imread(image_path)
    mask = imread(mask_path)

    # Normalize image
    img_min, img_max = np.percentile(image, (1, 99))
    image = exposure.rescale_intensity(image, in_range=(img_min, img_max))

    # Remove background with label value 1 from mask
    if args.background:
        mask[mask == args.background] = 0

    # Image info and Ray value initialization
    labels = np.unique(mask)[1:]
    n_labels = len(labels)
    chunk_size = args.num_chunks

    pending_futures = []
    refined_mask = np.zeros_like(mask)

    # Loops through Ray futures and processes chunk_size chunks at a time
    for i in range(0, n_labels, chunk_size):
        chunk_labels = labels[i:i + chunk_size]
        futures = [process_label.remote(mask, image, label) for label in chunk_labels if label != 0]
        pending_futures.extend(futures)

        if len(pending_futures) >= chunk_size:
            ready_futures, remaining_futures = ray.wait(pending_futures, num_returns=len(pending_futures))
            results = ray.get(ready_futures)
            refined_mask = update_mask(refined_mask, results)
            pending_futures = remaining_futures

    if pending_futures:
        ready_futures, _ = ray.wait(pending_futures, num_returns=len(pending_futures))
        results = ray.get(ready_futures)
        refined_mask = update_mask(refined_mask, results)

    # Save refined masks
    save_path = mask_path.parent / "refined_mask.tif"
    imwrite(save_path, refined_mask)


if __name__ == '__main__':
    main()
