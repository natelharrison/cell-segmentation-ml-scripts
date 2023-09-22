import argparse
import os
import sys

from typing import Tuple, Union, Optional

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from numpy import ndarray
from pathlib import Path
from tifffile import imread, imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='Path to the input image')
parser.add_argument('--mask_path', type=str, help='Path to the input mask')
parser.add_argument('--num_chunks', type=int, default=2,
                    help='Number of chunks to split image into along z-axis')
parser.add_argument('--background', type=int, help='Remove background if labeled')
parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization')
args = parser.parse_args()


def get_label_slice(mask: np.ndarray) -> ndarray[int]:
    """
    Get the index of the slice with the most prominent label in a 3D binary mask.
    :param mask: np.ndarray: The 3D binary mask.
    :return ndarray[int]: The index of the prominent slice.
    """
    slice_sums = mask.sum(axis=(1, 2))
    return np.argmax(slice_sums)


def mask_to_dask(mask: np.ndarray[np.uint8]) -> da.Array:
    labels = np.unique(mask)
    binary_mask = [
        (mask == label).astype(np.bool_) for label in labels
    ]
    return da.stack(binary_mask)


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


def get_bounding_box(
        binary_mask: np.ndarray, buffer: int = 16
) -> Tuple[int, int, int, int, int, int]:
    """
    Returns the bounding box for a given label
    :param binary_mask:
    :param buffer:
    :return bounding_coords:
    """
    x, y, z = np.where(binary_mask)

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


def process_label(
        binary_mask: np.ndarray,
        image: np.ndarray
) -> Optional[np.ndarray[np.bool_]]:
    """
    Process a label within a Pool of processes.
    :param binary_mask:
    :param image:
    :return active_contour():
    """

    if np.sum(binary_mask) <= 512:
        print(
            f"Label contains fewer than 100 labeled pixels - skipping"
        )
        sys.stdout.flush()
        return None

    x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(binary_mask)
    cropped_image = image[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_mask = binary_mask[x_min:x_max, y_min:y_max, z_min:z_max]

    refined_cropped_label = active_contour(cropped_image, cropped_mask)

    binary_mask[
        x_min: x_max, y_min: y_max, z_min: z_max
    ] = refined_cropped_label

    return binary_mask.astype(np.bool_)


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
        itk_image, sigma=2
    )

    # Geodesic active contour filter initialization
    img_filter = sitk.GeodesicActiveContourLevelSetImageFilter()
    img_filter.SetPropagationScaling(-2.0)
    img_filter.SetCurvatureScaling(10.0)
    img_filter.SetAdvectionScaling(10.0)
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
    return (sitk.GetArrayFromImage(refined_mask) > -1.0).astype(np.bool_)


def main():
    image_path: Union[str, Path] = Path(args.image_path)
    mask_path: Union[str, Path] = Path(args.mask_path)

    image: np.ndarray = imread(image_path)
    mask: np.ndarray = imread(mask_path)

    # Remove background with label value 1 from mask
    if args.background:
        mask[mask == args.background] = 0

    # Visualize each label as they are computed. Does not save outputs.
    # NOT WORKING WITH DASK, NEED TO FIX
    # if args.visualize:
    #     labels, pixel_count = np.unique(mask, return_counts=True)
    #     valid_labels = labels[np.where(pixel_count >= 1000)[0]]
    #     for label in valid_labels[1:]:
    #         _ = process_label((image, mask, label))
    #     return

    total_cpu_cores = os.cpu_count()
    cluster = LocalCluster(threads_per_worker=total_cpu_cores // 2)
    with Client(cluster) as client:

        mask_da = mask_to_dask(mask)
        refined_mask = np.zeros_like(mask)
        del mask

        with ProgressBar():
            results_da = mask_da.map_blocks(process_label, image)
        results = results_da.compute()

        label_counter = 1
        for binary_mask in results:
            refined_mask[binary_mask] = label_counter
            label_counter += 1

        # Save refined masks
        save_path = mask_path.parent / "refined_mask.tif"
        imwrite(save_path, refined_mask)



if __name__ == '__main__':
    main()
