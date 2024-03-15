import numpy as np
import cv2
from scipy import signal

def gaussian_kernel(size, sigma):
    """
    Generates a Gaussian kernel of the specified size and sigma.

    Parameters:
    size (int): The size of the kernel.
    sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
    numpy.ndarray: The generated Gaussian kernel.
    """

    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    kernel =  (1 / (2.0 * np.pi * sigma**2)) * np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
    
    return kernel


def sobel_filters(image):
    """
    Applies Sobel filters to the input image to compute the gradients in the x and y directions.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    tuple: A tuple containing the gradient images in the x and y directions.
    """

    # Sobel kernels
    Kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # Convolve the kernels with the image
    I_x = signal.convolve2d(image, Kernel_x, mode='same', boundary='wrap')
    I_y = signal.convolve2d(image, Kernel_y, mode='same', boundary='wrap')

    return I_x, I_y



def harris_corner_detection(img, k=0.06):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth the image using a Gaussian filter
    kernel = gaussian_kernel(5, 1)
    smoothed_image = signal.convolve2d(gray, kernel, mode='same', boundary='wrap')

    # Compute the image gradients using the Sobel filter
    Ix, Iy = sobel_filters(smoothed_image)

    # Compute the elements of the structure tensor
    M, N = gray.shape[:2]
    M_Matrix = np.zeros((M, N, 2, 2))
    M_Matrix[:, :, 0, 0] = np.square(Ix)
    M_Matrix[:, :, 0, 1] = Ix * Iy
    M_Matrix[:, :, 1, 0] = Iy * Ix
    M_Matrix[:, :, 1, 1] = np.square(Iy)

    # Compute the Harris corner response
    kernel = gaussian_kernel(5, 1)

    D = np.zeros_like(M_Matrix)
    for i in range(2):
        for j in range(2):
            D[:, :, i, j] = signal.convolve2d(M_Matrix[:, :, i, j], kernel, mode='same', boundary='symm')

    P = D[:, :, 0, 0]
    Q = D[:, :, 0, 1]
    R = D[:, :, 1, 1]

    lambda1 = ((P + R) / 2) - (np.sqrt(np.square(P - R) + 4 * np.square(Q)) / 2)
    lambda2 = ((P + R) / 2) + (np.sqrt(np.square(P - R) + 4 * np.square(Q)) / 2)

    HarrisResponses = lambda1 * lambda2 - k * np.square(lambda1 + lambda2)
    HarrisResponses = (HarrisResponses - HarrisResponses.min()) / (HarrisResponses.max() - HarrisResponses.min())

    # Threshold for corner detection
    corners_of_image = np.copy(img)
    corners_of_image[HarrisResponses > 0.48] = [255, 0, 0]

    return corners_of_image
