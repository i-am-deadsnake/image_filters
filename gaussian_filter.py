from itertools import product

from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros


def generate_gaussian_kernel(kernel_size, sigma):
    center = kernel_size // 2
    x, y = mgrid[0 - center : kernel_size - center, 0 - center : kernel_size - center]
    gaussian_kernel = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return gaussian_kernel


def apply_gaussian_filter(image, kernel_size, sigma):
    height, width = image.shape[0], image.shape[1]
    dst_height = height - kernel_size + 1
    dst_width = width - kernel_size + 1

    image_array = zeros((dst_height * dst_width, kernel_size * kernel_size))
    row = 0
    for i, j in product(range(dst_height), range(dst_width)):
        window = ravel(image[i : i + kernel_size, j : j + kernel_size])
        image_array[row, :] = window
        row += 1

    gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma)
    filter_array = ravel(gaussian_kernel)

    result = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

    return result


if __name__ == "__main__":
    image = imread(r"lena.jpg")
    gray_image = cvtColor(image, COLOR_BGR2GRAY)

    gaussian_3x3 = apply_gaussian_filter(gray_image, 3, sigma=1)
    gaussian_5x5 = apply_gaussian_filter(gray_image, 5, sigma=0.8)

    imshow("Gaussian filter 3x3", gaussian_3x3)
    imshow("Gaussian filter 5x5", gaussian_5x5)
    waitKey()

