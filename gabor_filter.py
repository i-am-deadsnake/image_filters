import numpy as np
from cv2 import COLOR_BGR2GRAY, CV_8UC3, cvtColor, filter2D, imread, imshow, waitKey


def generate_gabor_filter_kernel(
    kernel_size: int, sigma: int, theta: int, wavelength: int, gamma: int, psi: int
) -> np.ndarray:
    if (kernel_size % 2) == 0:
        kernel_size = kernel_size + 1
    gabor_filter = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for y in range(kernel_size):
        for x in range(kernel_size):
            px = x - kernel_size // 2
            py = y - kernel_size // 2

            theta_rad = theta / 180 * np.pi
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)

            _x = cos_theta * px + sin_theta * py

            _y = -sin_theta * px + cos_theta * py

            gabor_filter[y, x] = np.exp(
                -(_x**2 + gamma**2 * _y**2) / (2 * sigma**2)
            ) * np.cos(2 * np.pi * _x / wavelength + psi)

    return gabor_filter


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    image = imread("lena.jpg")
    gray_image = cvtColor(image, COLOR_BGR2GRAY)

    result = np.zeros(gray_image.shape[:2])
    for theta_value in [0, 30, 60, 90, 120, 150]:
        gabor_kernel_10 = generate_gabor_filter_kernel(10, 8, theta_value, 10, 0, 0)
        result += filter2D(gray_image, CV_8UC3, gabor_kernel_10)
    result = result / result.max() * 255
    result = result.astype(np.uint8)

    imshow("Original", gray_image)
    imshow("Gabor filter 20x20", result)

    waitKey(0)
