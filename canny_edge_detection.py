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
    Apply Sobel filter to the image to compute the gradients.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    tuple: A tuple containing the gradient magnitude and gradient direction arrays.
    """

    # Sobel kernels
    Kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # Convolve the kernels with the image
    I_x = signal.convolve2d(image, Kernel_x, mode='same', boundary='wrap')
    I_y = signal.convolve2d(image, Kernel_y, mode='same', boundary='wrap')

    # Compute the magnitude and direction of the gradients
    filtered_image = np.sqrt(I_x**2 + I_y**2)
    filtered_image = filtered_image / filtered_image.max() * 255
    theta = np.arctan2(I_y, I_x)

    return filtered_image, theta



def non_max_suppression(image, theta):
    """
    Applies non-maximum suppression to the input image based on the given gradient angles.

    Args:
        image (numpy.ndarray): The input image.
        theta (numpy.ndarray): The gradient angles of the image.

    Returns:
        numpy.ndarray: The image after non-maximum suppression.

    Description:
        Non-maximum suppression is a technique used in edge detection algorithms to thin out the detected edges.
        It works by suppressing pixels that are not local maxima in the direction of the gradient.
        The algorithm compares the intensity of a pixel with its neighbors in the direction of the gradient and
        keeps the pixel if it is the maximum among its neighbors. Otherwise, it sets the pixel intensity to zero.
    """

    height, width = image.shape
    m = np.zeros((height, width), dtype=np.int32)

    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for h in range(1, height-1):
        for w in range(1, width-1):

            if 0 <= angle[h, w] < 22.5 or 157.5 <= angle[h, w] <= 180:
                value = max(image[h, w-1], image[h, w+1])
            elif 22.5 <= angle[h, w] < 67.5:
                value = max(image[h-1, w+1], image[h+1, w-1])
            elif 67.5 <= angle[h, w] < 112.5:  
                value = max(image[h-1, w], image[h+1, w])
            elif 112.5 <= angle[h, w] < 157.5:  
                value = max(image[h-1, w-1], image[h+1, w+1])
  
            if image[h, w] >= value:
                m[h, w] = image[h, w]
            else:
                m[h, w] = 0

    m = np.multiply(m, 255/m.max())
    return m


def double_thresh(image, low_thresh, high_thresh):
    """
    Apply double threshold to the input image to detect strong, weak, and non-relevant pixels.

    Args:
        image (numpy.ndarray): The input image.
        low_thresh (int): The low threshold value.
        high_thresh (int): The high threshold value.

    Returns:
        numpy.ndarray: The image after applying double threshold.

    Description:
        Double thresholding is a technique used in edge detection algorithms to identify strong, weak, and non-relevant pixels.
        It works by classifying the pixels into one of the three categories based on their intensity values.
        The algorithm first identifies strong pixels, which are above the high threshold, and non-relevant pixels, which are below the low threshold.
        Then, it identifies weak pixels, which are between the low and high thresholds, and marks them for further processing.
    """

    strong = 255
    weak = 50

    strong_i, strong_j = np.where(image >= high_thresh)
    zeros_i, zeros_j = np.where(image < low_thresh)
    weak_i, weak_j = np.where((image <= high_thresh) & (image >= low_thresh))

    image[strong_i, strong_j] = strong
    image[zeros_i, zeros_j] = 0
    image[weak_i, weak_j] = weak

    return image

def hysteresis(image, weak, strong=255):

    """
    Apply hysteresis to the input image to detect edges.

    Args:
        image (numpy.ndarray): The input image.
        weak (int): The intensity value of weak pixels.
        strong (int): The intensity value of strong pixels.

    Returns:
        numpy.ndarray: The image after applying hysteresis.

    Description:
        Hysteresis is a technique used in edge detection algorithms to identify and connect weak edges to strong edges.
        It works by tracing along the strong edges and connecting them to weak edges if they are part of the same edge.
        The algorithm first identifies the strong edges and then traces along them to connect the weak edges to form complete edges.
    """

    height, width = image.shape

    for h in range(1, height-1):
        for w in range(1, width-1):
            if image[h, w] == weak:
                if ((image[h+1, w-1] == strong) or (image[h+1, w] == strong) or (image[h+1, w+1] == strong)
                    or (image[h, w-1] == strong) or (image[h, w+1] == strong)
                    or (image[h-1, w-1] == strong) or (image[h-1, w] == strong) or (image[h-1, w+1] == strong)):
                    image[h, w] = strong
                else:
                    image[h, w] = 0

    return image


def edge_detector(image, sigma, low_thresh, high_thresh):
    """
    Detects edges in an image using the Canny edge detection algorithm.

    Args:
        image (numpy.ndarray): The input image.
        sigma (float): The standard deviation of the Gaussian blur.
        low_thresh (float): The lower threshold for the double thresholding.
        high_thresh (float): The higher threshold for the double thresholding.

    Returns:
        numpy.ndarray: The edge image.

    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image to smooth the image
    kernel_size = 5
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed_image = signal.convolve2d(gray_image, kernel, mode='same', boundary='wrap')

    # Apply Sobel filter to the smoothed image to get the gradients
    filtered_image, theta = sobel_filters(smoothed_image)

    # Apply non-maximum suppression to the gradient magnitude
    suppressed_image = non_max_suppression(filtered_image, theta)

    # Apply double threshold to the suppressed image
    threshold_image = double_thresh(suppressed_image, low_thresh, high_thresh)

    # Apply hysteresis to the thresholded image to detect edges
    edge_image = hysteresis(threshold_image, weak=50, strong=255)

    return edge_image

