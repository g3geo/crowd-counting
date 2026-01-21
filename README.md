# 👥 Hology 8.0: Crowd Counting (Team Rosbloks)
> **Optimized CSRNet implementation for high-accuracy density estimation.**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)

## 📌 Project Overview
Repositori ini berisi solusi dari **Tim Rosbloks** untuk kompetisi **Penyisihan Hology 8.0 2025** kategori Data Mining. Proyek ini mengimplementasikan strategi *density regression* menggunakan arsitektur **CSRNet** yang dioptimalkan untuk menangani variasi skala dan kepadatan kerumunan yang ekstrem.

## 🚀 Key Features
* **CSRNet Architecture:** Menggunakan VGG16 sebagai *frontend* untuk ekstraksi fitur tingkat rendah dan *backend* berupa *stack dilated convolutions* untuk memperluas *receptive field*.
* **Adaptive Sigma Generation:** Pembuatan *density map* dinamis menggunakan filter Gaussian dengan sigma berbasis jarak $k$-Nearest Neighbors ($k=4$).
* **Stratified Density Splitting:** Membagi dataset menjadi 4 *bucket* kepadatan (sedikit, sedang, banyak, sangat banyak) untuk memastikan distribusi data *training* dan *validation* yang seimbang.
* **Density-Aware Weighted Loss:** Kombinasi MSE + MAE dengan pembobotan hingga 3x lebih besar untuk kasus kepadatan tinggi guna mengatasi *class imbalance*.
* **AMP Support:** Implementasi *Automatic Mixed Precision* untuk mempercepat *training* hingga 2x lebih cepat pada GPU.

## ⚙️ Model Configuration
Kami menggunakan konfigurasi yang telah disetel (*fine-tuned*) untuk stabilitas konvergensi:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Learning Rate** | $1 \times 10^{-5}$ | Conservative rate untuk mencegah osilasi gradien. |
| **Batch Size** | 4 | Optimal untuk memori GPU Kaggle dan kualitas gradien. |
| **Optimizer** | Adam | *Adaptive learning rate* yang tangguh. |
| **Resolution** | $768 \times 1024$ | Mempertahankan detail krusial pada gambar padat. |
| **Loss Function** | Weighted MSE+MAE | Menangani disparitas kepadatan antar sampel. |

## 📊 Training Performance
Model menunjukkan performa yang kompetitif dengan hasil eksperimen terakhir sebagai berikut:

* **Best Validation MAE:** $29.6071$ (Tercapai pada Epoch 18).
* **Target MAE Final:** $14 - 15$.
* **Training Time:** ~6 jam (100 Epochs) pada lingkungan GPU Kaggle.

## 🛠️ Execution Pipeline
1.  **Preprocessing:** Konversi koordinat kepala menjadi `.npy` *density map* adaptif.
2.  **Visualization:** Verifikasi statistik distribusi dan visualisasi *ground truth*.
3.  **Model Training:** Loop pelatihan utama dengan *gradient clipping* ($max\_norm=1.0$).
4.  **Inference:** Generasi file `submission.csv` dengan pembulatan integer dan limitasi nilai non-negatif.

---
**Team:** Rosbloks | **Competition:** Hology 8.0 2025
