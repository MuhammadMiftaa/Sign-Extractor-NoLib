import cv2
import numpy as np

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


img = cv2.imread("./input/raw.jpg", 0)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
result = sauvola_threshold(blurred)

cv2.imwrite("./output/result.png", result)