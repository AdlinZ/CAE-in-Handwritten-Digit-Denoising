import numpy as np
import os
from tensorflow.keras.datasets import mnist
from PIL import Image

# 1. 高斯噪声函数
def add_gaussian_noise(image, mean=0.0, sigma=0.1):
    """
    给输入的图像添加高斯噪声。

    参数:
        image: 输入图像，形状为 (28, 28)
        mean: 噪声均值
        sigma: 噪声标准差

    返回:
        带噪声的图像，范围限定在 [0, 255]
    """
    noise = np.random.normal(mean, sigma, image.shape)  # 生成高斯噪声
    noisy_image = np.clip(image + noise, 0.0, 1.0)  # 限制范围到 [0, 1]
    return (noisy_image * 255).astype(np.uint8)  # 还原到 [0, 255]，并转换为 uint8


# 2. 保存图像函数
def save_images(images, folder_path, prefix):
    """
    将输入图像保存到指定文件夹。

    参数:
        images: 图像数据列表，每个图像形状为 (28, 28)
        folder_path: 保存图像的文件夹路径
        prefix: 文件名前缀
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 创建文件夹

    for idx, img in enumerate(images):
        img = Image.fromarray(img.astype(np.uint8))  # 转换为 uint8 类型
        img.save(os.path.join(folder_path, f"{prefix}_{idx}.png"))

# 3. 主函数：加载 MNIST 并生成带噪声图像
def generate_noisy_mnist_images(save_path, num_samples=10, mean=0.0, sigma=0.1):
    """
    加载 MNIST 数据集，生成带高斯噪声的手写数字图像，并保存到本地。

    参数:
        save_path: 图像保存的文件夹路径
        num_samples: 生成带噪声图像的数量
        mean: 高斯噪声均值
        sigma: 高斯噪声标准差
    """
    # 加载 MNIST 数据集
    (train_images, _), (_, _) = mnist.load_data()
    train_images = train_images.astype('float32') / 255.0  # 归一化到 [0, 1]

    # 随机选择一些图像
    selected_images = train_images[:num_samples]

    # 添加噪声
    noisy_images = np.array([add_gaussian_noise(img) for img in selected_images])

    # 保存原始和带噪声的图像
    save_images(selected_images * 255, save_path, "original")  # 原始图像
    save_images(noisy_images, save_path, "noisy")  # 带噪声图像

    print(f"生成了 {num_samples} 张原始图像和带噪声的图像，保存在 '{save_path}' 文件夹中。")


# 4. 执行程序
if __name__ == "__main__":
    save_folder = "./noisy_mnist_images"  # 图像保存路径
    generate_noisy_mnist_images(save_path=save_folder, num_samples=20, sigma=0.2)
