from flask import Flask, request, render_template, send_file
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 设置上传文件夹
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# 加载去噪模型
model = load_model('denoising_autoencoder.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_and_denoise():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # 保存上传的文件
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # 读取图像并转换成灰度图像
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (28, 28))  # 假设模型处理28x28图像
        img_resized = img_resized / 255.0  # 归一化
        img_input = np.expand_dims(img_resized, axis=0)  # 添加batch维度
        img_input = np.expand_dims(img_input, axis=-1)  # 添加通道维度

        # 通过模型去噪
        denoised_img = model.predict(img_input)[0]
        denoised_img = denoised_img.squeeze() * 255.0  # 去掉多余维度并反归一化
        denoised_img = denoised_img.astype(np.uint8)

        # 保存去噪后的图像
        processed_path = os.path.join(PROCESSED_FOLDER, f"denoised_{file.filename}")
        cv2.imwrite(processed_path, denoised_img)

        # 显示原始与去噪后的图像
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img_resized, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(denoised_img, cmap='gray')
        plt.title("Denoised Image")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('comparison.png')  # 保存对比图像
        plt.close()

        # 返回去噪图像给用户
        return send_file('comparison.png', mimetype='image/png')

    return '''
    <!doctype html>
    <title>Upload Image for Denoising</title>
    <h1>Upload an Image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
