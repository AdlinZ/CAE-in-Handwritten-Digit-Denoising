import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 1. 定义去噪方法
def gaussian_filter_denoise(noisy_image, kernel_size=3):
    return cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)

def median_filter_denoise(noisy_image, kernel_size=3):
    return cv2.medianBlur(noisy_image, kernel_size)

def mean_filter_denoise(noisy_image, kernel_size=3):
    return cv2.blur(noisy_image, (kernel_size, kernel_size))

# 2. 载入数据并处理
(train_images, _), (test_images, _) = mnist.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 添加通道维度以适配CAE
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# 添加高斯噪声
def add_gaussian_noise(images, mean=0.0, sigma=0.2):
    noisy_images = images + np.random.normal(mean, sigma, images.shape)
    noisy_images = np.clip(noisy_images, 0.0, 1.0)  # 保证像素值在[0, 1]
    return noisy_images

test_noisy_images = add_gaussian_noise(test_images)

# 加载CAE模型
autoencoder = load_model('denoising_autoencoder.h5')

# 3. 定义计算PSNR和SSIM的函数
def calculate_metrics(original_images, denoised_images):
    psnr_values = []
    ssim_values = []
    for original, denoised in zip(original_images, denoised_images):
        psnr_value = psnr(original.squeeze(), denoised.squeeze(), data_range=1.0)
        ssim_value = ssim(original.squeeze(), denoised.squeeze(), data_range=1.0)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
    return np.mean(psnr_values), np.std(psnr_values), np.mean(ssim_values), np.std(ssim_values)

# 4. 对不同方法进行去噪并计算指标
methods = {
    "Gaussian Filter": lambda img: gaussian_filter_denoise((img * 255).astype(np.uint8)) / 255.0,
    "Median Filter": lambda img: median_filter_denoise((img * 255).astype(np.uint8)) / 255.0,
    "Mean Filter": lambda img: mean_filter_denoise((img * 255).astype(np.uint8)) / 255.0,
    "CAE": lambda img: autoencoder.predict(np.expand_dims(img, axis=0))[0]
}

results = []
for method_name, denoise_function in methods.items():
    denoised_images = np.array([denoise_function(img) for img in test_noisy_images])
    avg_psnr, std_psnr, avg_ssim, std_ssim = calculate_metrics(test_images, denoised_images)
    results.append((method_name, avg_psnr, std_psnr, avg_ssim, std_ssim))

# 5. 将结果汇总成表格
print(f"{'Method':<15} {'Avg PSNR':<10} {'PSNR Std':<10} {'Avg SSIM':<10} {'SSIM Std':<10}")
for method_name, avg_psnr, std_psnr, avg_ssim, std_ssim in results:
    print(f"{method_name:<15} {avg_psnr:<10.4f} {std_psnr:<10.4f} {avg_ssim:<10.4f} {std_ssim:<10.4f}")

# 6. 可视化去噪效果
sample_indices = [0, 1, 2]  # 选择3张样本进行可视化
sample_originals = test_images[sample_indices]
sample_noisy = test_noisy_images[sample_indices]

plt.figure(figsize=(15, len(sample_indices) * 5))

for i, idx in enumerate(sample_indices):
    plt.subplot(len(sample_indices), len(methods) + 2, i * (len(methods) + 2) + 1)
    plt.imshow(sample_originals[i].squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(len(sample_indices), len(methods) + 2, i * (len(methods) + 2) + 2)
    plt.imshow(sample_noisy[i].squeeze(), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    col = 3
    for method_name, denoise_function in methods.items():
        denoised_image = denoise_function(sample_noisy[i])
        plt.subplot(len(sample_indices), len(methods) + 2, i * (len(methods) + 2) + col)
        plt.imshow(denoised_image.squeeze(), cmap='gray')
        plt.title(method_name)
        plt.axis('off')
        col += 1

plt.tight_layout()
plt.show()
