import argparse
from multiprocessing import Pool

import itk
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from tifffile import imread, imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str)
parser.add_argument('--mask_path', type=str)
parser.add_argument('--remove_label', type=int)
args = parser.parse_args()


def get_label_slice(mask):
    slice_sums = mask.sum(axis=(1, 2))
    return np.argmax(slice_sums)


def visualize_3d_slice(image, mask, gradient_magnitude, refined_mask):
    # Determine the prominent slice for the mask
    slice_index = get_label_slice(mask)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Display the input image slice
    axes[0].imshow(image[slice_index], cmap='gray')
    axes[0].set_title('Input Image Slice')
    axes[0].axis('off')

    # Display the initial mask slice
    axes[1].imshow(mask[slice_index], cmap='gray')
    axes[1].set_title('Initial Mask Slice')
    axes[1].axis('off')

    # Display the gradient magnitude slice
    axes[2].imshow(gradient_magnitude[slice_index], cmap='jet')
    axes[2].set_title('Gradient Magnitude Slice')
    axes[2].axis('off')

    # Display the refined mask slice
    axes[3].imshow(refined_mask[slice_index], cmap='gray')
    axes[3].set_title('Refined Mask Slice')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


def process_label(data: tuple[np.ndarray, np.ndarray, int]):
    image, mask, label = data
    print(f"Processing mask {label}")
    label_mask = (mask == label).astype(np.uint16)
    return active_countour(image, label_mask), label


def active_countour(image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    itk_image = sitk.Cast(sitk.GetImageFromArray(image), sitk.sitkFloat32)
    itk_mask = sitk.Cast(sitk.GetImageFromArray(binary_mask), sitk.sitkFloat32)

    gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussian(
        itk_image, sigma=1.0
    )

    # Geodesic active contour filter initialization
    img_filter = sitk.GeodesicActiveContourLevelSetImageFilter()
    img_filter.SetPropagationScaling(1.0)
    img_filter.SetCurvatureScaling(1.0)
    img_filter.SetAdvectionScaling(1.0)
    img_filter.SetMaximumRMSError(0.01)
    img_filter.SetNumberOfIterations(1000)

    refined_mask = img_filter.Execute(itk_mask, gradient_magnitude)
    visualize_3d_slice(image, binary_mask, sitk.GetArrayFromImage(gradient_magnitude), sitk.GetArrayFromImage(refined_mask))
    return sitk.GetArrayFromImage(refined_mask)


def main():
    image_path = Path(args.image_path)
    mask_path = Path(args.mask_path)

    image: np.ndarray = imread(image_path.as_posix())
    mask: np.ndarray = imread(mask_path.as_posix())
    labels, pixel_count = np.unique(mask, return_counts=True)

    if labels[0] == 1 and pixel_count[0] >= 50000:
        mask[mask == 1] = 0
        print(f"Removed label {labels[0]} of size {pixel_count[0]}")

    refined_mask = np.zeros_like(mask)
    with Pool(processes=1) as pool:
        results = pool.map(
            process_label,
            [(image, mask, label) for label in labels[1:]]
        )

    for refined_label, label in results:
        refined_mask[refined_label > 0.5] = label

    save_path = mask_path.parent / "refined_mask.tif"
    imwrite(save_path, refined_mask)


if __name__ == '__main__':
    main()
