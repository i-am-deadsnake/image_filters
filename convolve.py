from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
from numpy import array, dot, pad, ravel, uint8, zeros


def image_to_columns(image, block_size):
    rows, cols = image.shape
    dst_height = cols - block_size[1] + 1
    dst_width = rows - block_size[0] + 1
    image_array = zeros((dst_height * dst_width, block_size[1] * block_size[0]))
    row = 0
    for i in range(0, dst_height):
        for j in range(0, dst_width):
            window = ravel(image[i : i + block_size[0], j : j + block_size[1]])
            image_array[row, :] = window
            row += 1

    return image_array


def image_convolution(image, filter_kernel):
    height, width = image.shape[0], image.shape[1]
    kernel_size = filter_kernel.shape[0]
    padding_size = kernel_size // 2
    image_padded = pad(image, padding_size, mode="edge")

    image_array = image_to_columns(image_padded, (kernel_size, kernel_size))

    kernel_array = ravel(filter_kernel)
    result = dot(image_array, kernel_array).reshape(height, width)
    return result


if __name__ == "__main__":
    image = imread(r"lena.jpg")
    gray_image = cvtColor(image, COLOR_BGR2GRAY)
    laplace_kernel = array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    output = image_convolution(gray_image, laplace_kernel).astype(uint8)
    imshow("Laplacian", output)
    waitKey(0)
