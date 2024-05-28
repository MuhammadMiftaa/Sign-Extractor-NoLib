import cv2
import numpy as np
import time


def sauvola_thresholding(img, window_size=25, k=0.2, R=128):
    """
    Menerapkan thresholding Sauvola pada gambar grayscale.

    Parameters:
    img (numpy.ndarray): Gambar input dalam mode grayscale.
    window_size (int): Ukuran jendela lokal (harus ganjil).
    k (float): Parameter k untuk metode Sauvola.
    R (int): Nilai dinamis untuk metode Sauvola.

    Returns:
    numpy.ndarray: Gambar biner hasil thresholding Sauvola.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size harus ganjil")

    # Batasi ukuran gambar untuk pengujian jika diperlukan
    # if img.shape[0] > 1000 or img.shape[1] > 1000:
    #     raise ValueError("Ukuran gambar terlalu besar untuk diproses dalam waktu yang wajar")


    # Membuat padding pada gambar untuk memudahkan perhitungan jendela di tepi
    pad_size = window_size // 2
    padded_img = np.pad(img, pad_size, mode='reflect')

    # Hasil thresholding Sauvola
    result = np.zeros_like(img)

    # Perulangan untuk setiap pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Mengambil jendela di sekitar pixel (i, j)
            window = padded_img[i:i + window_size, j:j + window_size]

            # Hitung rata-rata secara manual
            window_sum = 0
            for wi in range(window_size):
                for wj in range(window_size):
                    window_sum += window[wi, wj]
            mean = window_sum / (window_size * window_size)

            # Hitung standar deviasi secara manual
            variance_sum = 0
            for wi in range(window_size):
                for wj in range(window_size):
                    variance_sum += (window[wi, wj] - mean) ** 2
            stddev = (variance_sum / (window_size * window_size)) ** 0.5

            # Menghitung threshold Sauvola
            threshold = mean * (1 + k * (stddev / R - 1))

            # Terapkan threshold
            if img[i, j] > threshold:
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result

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

# Baca gambar
img = cv2.imread('./input/raw.jpg', 0)

start_time = time.time()
# Terapkan metode Sauvola
result = sauvola_thresholding(img)
end_time = time.time()
print(f"Waktu yang diperlukan: {end_time - start_time} detik")
cv2.imwrite("./result.png", result)

start_time = time.time()
# Terapkan metode Sauvola
result = sauvola_threshold(img)
end_time = time.time()
print(f"Waktu yang diperlukan (OpenCV): {end_time - start_time} detik")

# Simpan hasilnya
cv2.imwrite("./output/result.png", result)
