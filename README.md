# Sign Extractor
  Mendeteksi dan mengekstraksi tanda tangan untuk menghilangkan latar belakang. Hasil akhir berupa gambar dalam format .png

### Instalasi
  Jalankan perintah berikut untuk menginstall dependensi
```bash
pip install -r requirements.txt
```

### Diagram

![METODE SAUVOLA (2)](https://github.com/MuhammadMiftaa/Sign-Extractor-NoLib/assets/163877047/c43a83a2-12c6-488d-a603-b13d45f2551e)

### Input Gambar
Gambar dibaca dengan bantuan library cv2

### Sauvola Threshold
  ##### Padding  
  Membuat padding pada gambar untuk memudahkan perhitungan jendela
  ##### Ekstrak Jendela Setiap Pixel
  Untuk setiap pixel ditentukan jendela disekitarnya lalu diekstrak
  ##### Hitung Mean dan Standar Deviasi
  Rata-rata dan deviasi standar intensitas piksel dalam jendela dihitung 
  ##### Hitung Threshold
  Mean dan Standar Deviasi digunakan untuk menghitung Threshold Sauvola
  ##### Binarisasi Pixel
  Jika pixel melewati threshold maka pixel tersebut akan di-'set' menjadi putih (255). Jika belum melewati threshold maka akan menjadi hitam (0)

### Output
  Output berupa gambar tanda tangan tanpa latar belakang dalam format .png

# Pengembangan
  Sign Extractor dikembangkan lagi dengan menambah metode Gaussian Blur, Otsu Threshold dan PSNR untuk menilai hasil akhir program.

### Diagram
![METODE SAUVOLA (3)](https://github.com/MuhammadMiftaa/Sign-Extractor-NoLib/assets/163877047/13fad650-23ff-4c78-9ae1-eb55b72d9a68)

### Input Gambar
  Gambar dibaca dengan bantuan library cv2
### Blur Gaussian
 Setelah gambar dibaca, kernel size yang digunakan secara default berukuran (5,5). Ukuran ini akan digunakan nanti untuk menghitung kernel Gaussian.\
  Gambar tadi akan di beri padding dengan ukuran setengah dari kernel size untuk memudahkan penerapan kernel Gaussian nanti.\
  Selanjutnya kernel Gaussian akan dihitung dan diterapkan ke gambar. Hasilnya akan menghaluskan gambar dengan meratakan nilai piksel di sekitar lokal 
  yang ditentukan oleh kernel.
  
### Metode Sauvola 
  1. Membuat padding pada gambar untuk memudahkan perhitungan jendela
  2. Untuk setiap pixel ditentukan jendela disekitarnya lalu diekstrak
  3. Mean dan deviasi standar intensitas piksel dalam jendela dihitung 
  4. Mean dan Standar Deviasi digunakan untuk menghitung Threshold Sauvola
  5. Jika pixel melewati threshold maka pixel tersebut akan di-'set' menjadi putih (255). Jika belum melewati threshold maka akan menjadi hitam (0)
  
### Menghilangkan Noise
  Sebuah gambar biner 'blob' akan dihasilkan dengan menghitung mean dari semua pixel di gambar. Pixel dengan nilai diatas mean akan menjadi 'blob'.\
  Setiap blob ini akan diberi label numerik sendiri. Lalu blob yang memiliki luas lebih dari 11 pixel akan ditotal semua nilai nya dan dihitung meannya.\
  Nilai mean ini akan digunakan untuk menentukan threshold. Blob yang masuk kategori kecil akan dihapus.
  
### Metode Otsu
  Histogram intensitas pixel gambar akan dihitung terlebih dahulu.\
  Histogram ini akan digunakan untuk menghitung probabilitas untuk membantu memisahkan foreground dan background.\
  Selanjutnya varian intra-kelas akan dihitung. Nilai ini akan dimaksimalkan untuk memisahkan foreground dan background.\
  Iterasi dilakukan untuk setiap level intensitas (0-255) dan akan dihitung nilai varian intra-kelasnya.\
  Nilai level intensitas yang memaksimalkan nilai varian intra-kelas akan dipilih menjadi threshold.\
  Akhirnya threshold ini diterapkan ke gambar. Pixel yang melewati nilai threshold akan dibuat menjadi foreground/putih (255) dan yang dibawah nilai threshold menjadi background/hitam (0).\
  Lalu gambar akhir merupakan gambar yang diinversikan.
  
### PSNR, Output
  Output berupa hasil akhir dari Metode Otsu. Output dapat diuji dengan menggunakan nilai PSNR.
