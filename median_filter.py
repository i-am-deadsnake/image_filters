from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
from numpy import divide, int8, multiply, ravel, sort, zeros_like


def apply_median_filter(gray_image, mask_size=3):
    border_size = int(mask_size / 2)
    median_image = zeros_like(gray_image)
    
    for i in range(border_size, gray_image.shape[0] - border_size):
        for j in range(border_size, gray_image.shape[1] - border_size):
            kernel = ravel(gray_image[i - border_size : i + border_size + 1, j - border_size : j + border_size + 1])
            median = sort(kernel)[int8(divide((multiply(mask_size, mask_size)), 2) + 1)]
            median_image[i, j] = median
            
    return median_image


if __name__ == "__main__":
    image = imread("lena.jpg")
    gray_image = cvtColor(image, COLOR_BGR2GRAY)

    median_3x3 = apply_median_filter(gray_image, 3)
    median_5x5 = apply_median_filter(gray_image, 5)

    imshow("Median filter 3x3", median_3x3)
    imshow("Median filter 5x5", median_5x5)
    waitKey(0)
