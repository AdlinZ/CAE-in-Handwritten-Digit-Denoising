卷积自动编码器在手写数字去噪中的应用研究
项目简介
本项目旨在通过 卷积自动编码器（CAE） 模型，实现手写数字图像的去噪任务。研究过程中使用了 MNIST 数据集 并引入高斯噪声，对模型在不同噪声水平下的表现进行了验证和评估。

项目结构

项目根目录
│── noisy_mnist_images/           # 包含添加噪声后的 MNIST 数据集
│── processed/                    # 处理后的图像文件
│── uploads/                      # 其他上传的辅助文件
│
│── denoising_autoencoder.h5      # 训练好的模型权重文件
│── denoising_autoencoder.ipynb   # Jupyter Notebook 实验代码
│
│── autoencoder_model.png         # 模型结构可视化图
│── autoencoder_model_horizontal.png
│
│── app_web_interface.py          # 可视化界面
│── generate_noisy_images.py      # 生成噪声数据的脚本
│── denoising_autoencoder.py      # 自动编码器模型训练与评估代码
│── compare_diff_model.py         # 不同模型效果对比代码
│── calculate_diff_sigma.py       # 噪声水平计算脚本
│── model_draw.py                 # 绘制模型结构脚本
│── calculate_all.py              # 自动化计算脚本
│
│── 卷积自动编码器在手写数字去噪中的应用研究.pdf    # 研究报告（PDF 版）
环境配置与依赖
运行环境要求
Python 版本：Python 3.8 及以上
深度学习框架：TensorFlow 2.x（当前代码基于 TensorFlow）
依赖库：
numpy
matplotlib
tensorflow
scikit-image
tqdm
安装依赖
请使用以下命令快速安装依赖：

pip install numpy matplotlib tensorflow scikit-image tqdm
快速开始
1. 生成带噪声的 MNIST 数据集
运行 generate_noisy_images.py，生成不同高斯噪声水平下的 MNIST 数据集：

python generate_noisy_images.py
2. 训练与评估自动编码器模型
运行 denoising_autoencoder.py，训练自动编码器模型，并进行噪声去除任务：

python denoising_autoencoder.py
3. 结果可视化
使用 denoising_autoencoder.ipynb Notebook 文件，可以直观查看去噪效果和不同噪声水平下的重建性能。


实验结果与分析
通过 CAE 模型，成功去除了不同水平的高斯噪声，并保持了图像的细节与结构。具体结果见以下部分：

未来工作
优化网络结构：探索更深层次的网络或混合模型，以提升去噪性能。
拓展任务场景：研究 CAE 在彩色图像去噪和医学图像处理中的应用潜力。
可视化界面：开发基于 Flask / Streamlit 的前端界面，供用户在线体验模型去噪效果。
