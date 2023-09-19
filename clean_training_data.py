import argparse
from multiprocessing import Pool
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage.measure

from numpy import ndarray
from pathlib import Path
from tifffile import imread, imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='Path to the input image')
parser.add_argument('--mask_path', type=str, help='Path to the input mask')
parser.add_argument('--remove_label', type=int, help='Label to be removed from the mask')
parser.add_argument('--processes', type=int, help='Number of processes for parallel processing')
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


def visualize_3d_slice(
        image: np.ndarray,
        gradient_magnitude: np.ndarray,
        mask: np.ndarray,
        refined_mask: np.ndarray,
):
    """
    Visualize slices of input image, mask, gradient magnitude, and refined mask.

    :param gradient_magnitude: The initial mask.
    :param refined_mask: The gradient magnitude.
    :param image: The input image.
    :param mask: The refined mask.
    """
    slice_index = get_label_slice(gradient_magnitude)

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
        mask: np.ndarray, label: int, buffer: int = 16
) -> Tuple[int, int, int, int, int, int]:
    """
    Returns the bounding box for a given label
    :param mask:
    :param label:
    :param buffer:
    :return bounding_coords:
    """
    props = skimage.measure.regionprops((mask == label).astype(int))

    min_x, min_y, min_z, max_x, max_y, max_z = props[0].bbox

    return (
        max(0, min_x - buffer),
        min(mask.shape[0], max_x + buffer),
        max(0, min_y - buffer),
        min(mask.shape[1], max_y + buffer),
        max(0, min_z - buffer),
        min(mask.shape[2], max_z + buffer)
    )


def process_label(
        data: Tuple[np.ndarray, np.ndarray, int]
) -> Tuple[np.ndarray, int]:
    """
    Process a label within a Pool of processes.
    :param data:
    :return active_contour(), label:
    """
    image, mask, label = data
    print(f"Processing mask {label}")
    label_mask = (mask == label).astype(np.uint8)

    x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(mask, label)
    cropped_image = image[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_mask = label_mask[x_min:x_max, y_min:y_max, z_min:z_max]

    refined_cropped_mask = active_contour(cropped_image, cropped_mask)

    label_mask[
        x_min: x_max, y_min: y_max, z_min: z_max
    ] = refined_cropped_mask

    return label_mask, label


def process_labels(
        data: Tuple[np.ndarray, np.ndarray, int]
) -> Tuple[np.ndarray, int]:
    """
    Process a label within a Pool of processes.

    Parameters:
    - data (Tuple[np.ndarray, np.ndarray, int]): A tuple containing image, mask, and label.

    Returns:
    Tuple[np.ndarray, int]: A tuple containing the refined mask and label.
    """
    image, mask, label = data
    print(f"Processing mask {label}")
    label_mask = (mask == label).astype(np.uint8)
    return active_contour(image, label_mask), label


def active_contour(
        image: np.ndarray,
        binary_mask: np.ndarray
) -> np.ndarray[np.uint8]:
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
    img_filter.SetPropagationScaling(-4.0)
    img_filter.SetCurvatureScaling(10.0)
    img_filter.SetAdvectionScaling(10.0)
    img_filter.SetMaximumRMSError(0.01)
    img_filter.SetNumberOfIterations(250)

    refined_mask = img_filter.Execute(itk_mask, gradient_magnitude)

    if args.visualize:
        visualize_3d_slice(
            image,
            sitk.GetArrayFromImage(gradient_magnitude),
            binary_mask,
            sitk.GetArrayFromImage(refined_mask).astype(np.float16)
        )

    # Convert the refined mask into binary
    return (sitk.GetArrayFromImage(refined_mask) > -1.0).astype(np.uint8)


def main():
    image_path: Union[str, Path] = Path(args.image_path)
    mask_path: Union[str, Path] = Path(args.mask_path)

    image: np.ndarray = imread(image_path)
    mask: np.ndarray = imread(mask_path)
    labels, pixel_count = np.unique(mask, return_counts=True)

    # Set background to 0 if labeled
    if labels[0] == 1 and pixel_count[0] >= 50000:
        mask[mask == 1] = 0
        print(f"Removed label {labels[0]} of size {pixel_count[0]}")

    # Refine masks using active contour
    refined_mask = np.zeros_like(mask)
    with Pool(processes=args.processes) as pool:
        results = pool.map(
            process_label,
            [(image, mask, label) for label in labels[1:]]
        )

    for refined_label, label in results:
        refined_mask[np.where(refined_label == 1)] = label

    # Save refined masks
    save_path = mask_path.parent / "refined_mask.tif"
    imwrite(save_path, refined_mask)


if __name__ == '__main__':
    main()
