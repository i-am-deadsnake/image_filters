import math
import sys

import cv2
import numpy as np


def gaussian_function(matrix: np.ndarray, variance: float) -> np.ndarray:
    sigma = math.sqrt(variance)
    constant = 1 / (sigma * math.sqrt(2 * math.pi))
    return constant * np.exp(-((matrix / sigma) ** 2) * 0.5)


def get_image_slice(image: np.ndarray, x: int, y: int, kernel_size: int) -> np.ndarray:
    half_kernel = kernel_size // 2
    return image[x - half_kernel : x + half_kernel + 1, y - half_kernel : y + half_kernel + 1]


def generate_gaussian_kernel(kernel_size: int, spatial_variance: float) -> np.ndarray:
    distance_matrix = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance_matrix[i, j] = math.sqrt(
                abs(i - kernel_size // 2) ** 2 + abs(j - kernel_size // 2) ** 2
            )
    return gaussian_function(distance_matrix, spatial_variance)


def apply_bilateral_filter(
    input_image: np.ndarray,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int,
) -> np.ndarray:
    output_image = np.zeros_like(input_image, dtype=float)
    gaussian_kernel = generate_gaussian_kernel(kernel_size, spatial_variance)
    size_x, size_y = input_image.shape
    for i in range(kernel_size // 2, size_x - kernel_size // 2):
        for j in range(kernel_size // 2, size_y - kernel_size // 2):
            image_slice = get_image_slice(input_image, i, j, kernel_size)
            intensity_difference = image_slice - image_slice[kernel_size // 2, kernel_size // 2]
            intensity_weights = gaussian_function(intensity_difference, intensity_variance)
            spatial_weights = np.multiply(gaussian_kernel, intensity_weights)
            weighted_values = np.multiply(image_slice, spatial_weights)
            filtered_value = np.sum(weighted_values) / np.sum(spatial_weights)
            output_image[i, j] = filtered_value
    return output_image


def parse_command_line_args(args: list) -> tuple:
    input_filename = args[1] if len(args) > 1 else "lena.jpg"
    spatial_variance = float(args[2]) if len(args) > 2 else 1.0
    intensity_variance = float(args[3]) if len(args) > 3 else 1.0
    kernel_size = int(args[4]) if len(args) > 4 else 5
    kernel_size = kernel_size + abs(kernel_size % 2 - 1)  # Ensure kernel_size is odd
    return input_filename, spatial_variance, intensity_variance, kernel_size


if __name__ == "__main__":
    input_filename, spatial_variance, intensity_variance, kernel_size = parse_command_line_args(sys.argv)
    input_image = cv2.imread(input_filename, 0)
    cv2.imshow("Input Image", input_image)

    normalized_image = input_image / 255.0
    normalized_image = normalized_image.astype("float32")
    filtered_image = apply_bilateral_filter(normalized_image, spatial_variance, intensity_variance, kernel_size)
    filtered_image = filtered_image * 255
    filtered_image = np.uint8(filtered_image)
    cv2.imshow("Output Image", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
