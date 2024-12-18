import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 1. 加载 MNIST 数据集
(_, _), (test_images, _) = mnist.load_data()
test_images = test_images.astype('float32') / 255.0  # 归一化到 [0, 1]
test_images = np.expand_dims(test_images, axis=-1)  # 扩展维度 (28, 28, 1)

# 2. 加载训练好的去噪模型
model = load_model('denoising_autoencoder.h5')

# 3. 添加高斯噪声函数
def add_gaussian_noise(images, mean=0.0, sigma=0.1):
    """
    为输入图像添加高斯噪声
    """
    noisy_images = []
    for img in images:
        noise = np.random.normal(mean, sigma, img.shape)
        noisy_image = np.clip(img + noise, 0.0, 1.0)  # 确保像素值范围在 [0, 1]
        noisy_images.append(noisy_image)
    return np.array(noisy_images)

# 4. 测试不同噪声强度
noise_levels = [0.1, 0.2, 0.3]  # 不同的噪声强度
results = []

for sigma in noise_levels:
    # 添加高斯噪声
    test_images_noisy = add_gaussian_noise(test_images, sigma=sigma)

    # 使用模型去噪
    denoised_images = model.predict(test_images_noisy)

    # 计算PSNR和SSIM
    psnr_values = []
    ssim_values = []
    for i in range(len(test_images)):
        original = test_images[i].squeeze()
        denoised = denoised_images[i].squeeze()

        # 计算PSNR和SSIM
        psnr_value = psnr(original, denoised, data_range=1.0)
        ssim_value = ssim(original, denoised, data_range=1.0)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    # 计算平均PSNR和SSIM
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    results.append((sigma, mean_psnr, mean_ssim))

# 5. 打印结果表格
print("噪声强度 (σ)\t平均PSNR\t平均SSIM")
for sigma, mean_psnr, mean_ssim in results:
    print(f"{sigma:.1f}\t\t{mean_psnr:.4f}\t{mean_ssim:.4f}")
