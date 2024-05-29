import cv2
from skimage import measure
from skimage.measure import regionprops
import numpy as np
import time
from math import log10, sqrt 

def simple_threshold(img, threshold=128):
    """
    Fungsi untuk melakukan thresholding sederhana.
    
    Parameters:
    img (numpy.ndarray): Gambar input.
    threshold (int): Nilai threshold. Default adalah 128.
    
    Returns:
    numpy.ndarray: Gambar biner hasil thresholding.
    """
    # Terapkan threshold
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

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

def otsu_threshold(image):
    # Hitung histogram
    histogram = np.bincount(image.ravel(), minlength=256)
    
    # Hitung probabilitas
    total = image.size
    current_max, threshold = 0, 0
    sum_total, sum_foreground, weight_background, weight_foreground = 0, 0, 0, 0
    
    for i in range(256):
        sum_total += i * histogram[i]
    
    for i in range(256):
        weight_background += histogram[i]
        if weight_background == 0:
            continue
        
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        
        sum_foreground += i * histogram[i]
        
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground
        
        # Hitung varians intra-kelas
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i
    
    # Terapkan threshold dan inversi
    binary_img = (image > threshold).astype(np.uint8) * 255
    inverted_binary_img = 255 - binary_img
    
    return inverted_binary_img

def GaussianBlur(img, kernel_size=(5, 5)):
    """
    Melakukan Gaussian Blur pada gambar menggunakan implementasi manual.
    
    Parameters:
    img (numpy.ndarray): Gambar input.
    kernel_size (tuple): Ukuran kernel Gaussian Blur. Default adalah (5, 5).
    
    Returns:
    numpy.ndarray: Gambar hasil Gaussian Blur.
    """
    rows, cols = img.shape
    k_rows, k_cols = kernel_size

    padded_img = np.pad(img, ((k_rows//2, k_rows//2), (k_cols//2, k_cols//2)), mode='constant')

    blurred_img = np.zeros_like(img)
    kernel = gaussian_kernel(kernel_size)

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

def clearOtherComponent(img, pk_1=84,pk_2=250,pk_3=100,pk_4=37):
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)

    total_area = 0
    counter = 0
    rata_rata = 0.0

    for region in regionprops(blobs_labels):
        if (region.area > 11):
            total_area = total_area + region.area
            counter = counter + 1
    rata_rata = (total_area/counter)

    small_threshold = ((rata_rata/pk_1)*pk_2)+pk_3
    big_threshold = small_threshold*pk_4

    pre_version = morphology_remove_small_objects(blobs_labels, small_threshold)
    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > (big_threshold)
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0

    cv2.imwrite("./preversion.png", pre_version)

def morphology_remove_small_objects(blobs_labels, min_size):
    # Buat salinan untuk memastikan tidak ada perubahan pada input asli
    labels_out = np.copy(blobs_labels)
    num_labels = np.max(labels_out)
    
    for label in range(1, num_labels + 1):
        if np.sum(labels_out == label) < min_size:
            labels_out[labels_out == label] = 0
    
    return labels_out

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def PSNRread(input,output): 
     original = cv2.imread(input) 
     compressed = cv2.imread(output, 1) 
     value = PSNR(original, compressed) 
     print(f"Nilai PSNR: {value} dB") 

input_path = "./input/raw3.jpg"
output_path = "./output/result.png"

img = cv2.imread(input_path, 0)

start_time = time.time()

blurred = GaussianBlur(img)
binary_img = sauvola_threshold(blurred)
clearOtherComponent(binary_img)
cleaned_img = cv2.imread("./preversion.png", 0)
result = otsu_threshold(cleaned_img)

end_time = time.time()
print(f"Waktu yang diperlukan (OpenCV): {end_time - start_time} detik")

cv2.imwrite(output_path, result)

PSNRread(input_path,output_path)