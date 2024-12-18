import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 1. 加载 MNIST 数据集
(train_images, _), (_, _) = mnist.load_data()
train_images = train_images.astype('float32') / 255.0  # 归一化到 [0, 1]
train_images = np.expand_dims(train_images, axis=-1)  # 扩展维度 (28, 28, 1)

# 2. 划分训练集和验证集
X_train, X_val = train_test_split(train_images, test_size=0.2, random_state=42)

# 3. 添加高斯噪声函数
def add_gaussian_noise(image, mean=0.0, sigma=0.1):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0.0, 1.0)  # 保证像素值范围在 [0, 1]
    return noisy_image

# 4. 添加噪声到训练集和验证集
X_train_noisy = np.array([add_gaussian_noise(img) for img in X_train])
X_val_noisy = np.array([add_gaussian_noise(img) for img in X_val])

# 5. 检查带噪声和原始数据范围
print("带噪声图像范围:", X_train_noisy.min(), X_train_noisy.max())
print("原始图像范围:", X_train.min(), X_train.max())

# 6. 构建去噪自动编码器模型
input_img = Input(shape=(28, 28, 1))

# 编码器
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 解码器
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# 定义模型
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 7. 训练模型
autoencoder.fit(X_train_noisy, X_train, epochs=10, batch_size=128,
                validation_data=(X_val_noisy, X_val))

# 8. 测试并可视化结果
decoded_imgs = autoencoder.predict(X_val_noisy[:10])

# 定义计算 PSNR 和 SSIM 的函数
def calculate_metrics(original, denoised):
    psnr_value = psnr(original, denoised, data_range=1.0)
    ssim_value = ssim(original, denoised, data_range=1.0)
    return psnr_value, ssim_value
    
# 显示原始、带噪声和去噪后的图像，同时在去噪图像上标注 PSNR 和 SSIM
n = 5  # 显示5个样本
plt.figure(figsize=(15, 6))
for i in range(n):
    # 带噪声图像
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_val_noisy[i].reshape(28, 28), cmap='gray')
    plt.title("Noisy Image")
    plt.axis('off')

    # 原始图像
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(X_val[i].reshape(28, 28), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # 去噪后的图像
    denoised_image = decoded_imgs[i].reshape(28, 28)
    psnr_value, ssim_value = calculate_metrics(X_val[i].reshape(28, 28), denoised_image)

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_image, cmap='gray')
    plt.title(f"PSNR: {psnr_value:.2f}\nSSIM: {ssim_value:.4f}")  # 标注 PSNR 和 SSIM
    plt.axis('off')

plt.tight_layout()
plt.show()

# 保存模型
autoencoder.save('denoising_autoencoder.h5')
print("Model saved successfully!")

from tensorflow.keras.utils import plot_model

