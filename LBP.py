import cv2
import numpy as np
import matplotlib.pyplot as plt

# LBP特征提取函数
def compute_lbp(image):
    radius = 1
    num_points = 8
    lbp = np.zeros_like(image)  # 创建与输入图像相同大小的空白图像
    height, width = image.shape[:2]

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = image[y, x]  # 中心像素值
            code = 0  # LBP编码
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points  # 均匀采样的角度
                x_i = x + int(radius * np.cos(angle))  # 均匀采样点的x坐标
                y_i = y + int(radius * np.sin(angle))  # 均匀采样点的y坐标
                if image[y_i, x_i] >= center:
                    code |= (1 << i)  # 根据采样点的像素值决定编码位的值
            lbp[y, x] = code
    return lbp
# 加载图像并转换为灰度图像
image1 = cv2.imread('lane1.jpg', 0)
image2 = cv2.imread('lane2.jpg', 0)
# image2 = cv2.imread('img.jpg', 0)


# 提取LBP特征
lbp1 = compute_lbp(image1)
lbp2 = compute_lbp(image2)

print(lbp1,"\n",lbp2,"\n")

# 生成直方图
hist1, _ = np.histogram(lbp1.ravel(), bins=256, range=[0, 256])
hist2, _ = np.histogram(lbp2.ravel(), bins=256, range=[0, 256])

# 绘制直方图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist1)
plt.title('LBP Histogram - lane1.jpg')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.plot(hist2)
plt.title('LBP Histogram - lane2.jpg')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# # 比较特征差异
chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2))

print('卡方统计量:', chi_square)
