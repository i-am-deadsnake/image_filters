import numpy as np
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey

from digital_image_processing.filters.convolve import apply_image_convolution


def apply_sobel_filter(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    result_x = np.abs(apply_image_convolution(image, kernel_x))
    result_y = np.abs(apply_image_convolution(image, kernel_y))
    result_x = result_x * 255 / np.max(result_x)
    result_y = result_y * 255 / np.max(result_y)

    result_xy = np.sqrt((np.square(result_x)) + (np.square(result_y)))
    result_xy = result_xy * 255 / np.max(result_xy)
    result = result_xy.astype(np.uint8)

    theta = np.arctan2(result_y, result_x)
    return result, theta


if __name__ == "__main__":
    input_image = imread("lena.jpg")
    gray_image = cvtColor(input_image, COLOR_BGR2GRAY)

    sobel_gradient, sobel_theta = apply_sobel_filter(gray_image)

    imshow("Sobel filter", sobel_gradient)
    imshow("Sobel theta", sobel_theta)
    waitKey(0)

