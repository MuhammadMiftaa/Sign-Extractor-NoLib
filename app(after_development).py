import cv2
import numpy as np
import time

def sauvola_threshold(img, window_size=25, k=0.2, R=128):
    if window_size % 2 == 0:
        raise ValueError("window_size harus ganjil")

    # Membuat padding pada gambar untuk memudahkan perhitungan jendela di tepi
    pad_size = window_size // 2
    padded_img = np.pad(img, pad_size, mode='reflect')

    # Hitung rata-rata menggunakan konvolusi
    kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
    mean = cv2.filter2D(padded_img, -1, kernel)[pad_size:-pad_size, pad_size:-pad_size]

    # Hitung standar deviasi menggunakan konvolusi
    mean_sq = cv2.filter2D(padded_img ** 2, -1, kernel)[pad_size:-pad_size, pad_size:-pad_size]
    stddev = np.sqrt(mean_sq - mean ** 2)

    # Hitung threshold Sauvola
    threshold = mean * (1 + k * (stddev / R - 1))

    # Terapkan threshold
    result = np.where(img > threshold, 255, 0).astype(np.uint8)

    return result

def GaussianBlur(img, kernel_size=(5, 5)):
    """
    Melakukan Gaussian Blur pada gambar menggunakan implementasi manual.

    Parameters:
    img (numpy.ndarray): Gambar input.
    kernel_size (tuple): Ukuran kernel Gaussian Blur. Default adalah (5, 5).

    Returns:
    numpy.ndarray: Gambar hasil Gaussian Blur.
    """
    # Mendapatkan ukuran gambar
    rows, cols = img.shape

    # Mendapatkan ukuran kernel
    k_rows, k_cols = kernel_size

    # Padding gambar
    pad_rows = rows + k_rows - 1
    pad_cols = cols + k_cols - 1
    padded_img = np.pad(img, ((k_rows//2, k_rows//2), (k_cols//2, k_cols//2)), mode='constant')

    # Inisialisasi gambar hasil
    blurred_img = np.zeros_like(img)

    # Kernel Gaussian
    kernel = gaussian_kernel(kernel_size)

    # Konvolusi
    for i in range(rows):
        for j in range(cols):
            blurred_img[i, j] = np.sum(padded_img[i:i+k_rows, j:j+k_cols] * kernel)

    return blurred_img.astype(np.uint8)

def gaussian_kernel(kernel_size, sigma=1):
    """
    Menghasilkan kernel Gaussian.

    Parameters:
    kernel_size (tuple): Ukuran kernel Gaussian.
    sigma (float): Nilai sigma untuk Gaussian. Default adalah 1.

    Returns:
    numpy.ndarray: Kernel Gaussian.
    """
    rows, cols = kernel_size
    kernel = np.zeros(kernel_size)
    center_row, center_col = rows // 2, cols // 2

    for i in range(rows):
        for j in range(cols):
            kernel[i, j] = np.exp(-((i - center_row) ** 2 + (j - center_col) ** 2) / (2 * sigma ** 2))

    return kernel / np.sum(kernel)


img = cv2.imread("./input/raw.jpg", 0)

blurred = GaussianBlur(img)

result = sauvola_threshold(blurred)



cv2.imwrite("./output/result.png", result)