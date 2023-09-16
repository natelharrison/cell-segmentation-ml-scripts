import argparse

import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tifffile import imread, imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str)
parser.add_argument('--mask_path', type=str)
parser.add_argument('--remove_label', type=int)
args = parser.parse_args()


def active_countour(image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(binary_mask)

    gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussian(
        image, sigma=1.0
    )

    # Geodesic active contour filter initialization
    img_filter = sitk.GeodesicActiveContourLevelSetImageFilter()
    img_filter.SetPropagationScaling(1.0)
    img_filter.SetCurvatureScaling(1.0)
    img_filter.SetAdvectionScaling(1.0)
    img_filter.SetMaximumRMSError(0.01)
    img_filter.SetNumberOfIterations(1000)

    refined_mask = img_filter.Execute(mask, gradient_magnitude)
    return sitk.GetArrayFromImage(refined_mask)


def main():
    image_path = Path(args.image_path)
    mask_path = Path(args.mask_path)

    image = imread(image_path.as_posix())
    mask = imread(mask_path.as_posix())
    labels, pixel_count = np.unique(mask, return_counts=True)

    if labels[0] == 1 and pixel_count >= 50000:
        mask = mask[mask == 1] = 0
        print(f"Removed label {labels[0]} of size {pixel_count[0]}")

    refined_mask = np.zeros_like(mask)
    for label in labels[1:]:
        label_mask = (mask == label)

        refined_label = active_countour(image, label_mask)

        refined_mask[refined_label > 0.5] = label

    imwrite(mask_path, refined_mask)


if __name__ == '__main__':
    main()
