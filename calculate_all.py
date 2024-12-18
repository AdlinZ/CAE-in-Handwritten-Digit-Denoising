import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 1. 加载 MNIST 数据集
(train_images, _), (test_images, _) = mnist.load_data()
test_images = test_images.astype('float32') / 255.0  # 归一化到 [0, 1]
test_images = np.expand_dims(test_images, axis=-1)  # 扩展维度 (28, 28, 1)

# 2. 加载训练好的去噪模型
model = load_model('denoising_autoencoder.h5')

# 3. 添加高斯噪声函数
def add_gaussian_noise(image, mean=0.0, sigma=0.2):
    """
    添加高斯噪声到图像上
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0.0, 1.0)  # 确保像素值范围在 [0, 1]
    return noisy_image

# 4. 构建带噪声的测试集
test_images_noisy = np.array([add_gaussian_noise(img) for img in test_images])

# 5. 使用模型去噪
denoised_images = model.predict(test_images_noisy)

# 6. 计算PSNR和SSIM
psnr_values = []
ssim_values = []

for i in range(len(test_images)):
    original = test_images[i].squeeze()  # 去掉多余维度 (28, 28, 1) -> (28, 28)
    noisy = test_images_noisy[i].squeeze()
    denoised = denoised_images[i].squeeze()

    # 计算PSNR
    psnr_value = psnr(original, denoised, data_range=1.0)
    psnr_values.append(psnr_value)

    # 计算SSIM
    ssim_value = ssim(original, denoised, data_range=1.0)
    ssim_values.append(ssim_value)

# 7. 统计平均值和标准差
mean_psnr = np.mean(psnr_values)
std_psnr = np.std(psnr_values)
mean_ssim = np.mean(ssim_values)
std_ssim = np.std(ssim_values)

# 8. 打印结果
print("样本数:", len(test_images))
print("平均PSNR:", mean_psnr)
print("PSNR标准差:", std_psnr)
print("平均SSIM:", mean_ssim)
print("SSIM标准差:", std_ssim)
