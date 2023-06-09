# coding: utf-8
import cv2 as cv
from skimage import feature as skif
import numpy as np
import matplotlib.pyplot as plt


#获取图像的lbp特征
def get_lbp_data(image_path, lbp_radius=1, lbp_point=8):
    # img = utils.change_image_rgb(image_path)
    img = cv.imread(image_path)
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 使用LBP方法提取图像的纹理特征.
    #lbp_point：选取中心像素周围的像素点的个数；lbp_radius：选取的区域的半径
    #以下为5种不同的方法提取的lbp特征，相应的提取到的特征维度也不一样
    #'default': original local binary pattern which is gray scale but notrotation invariant
    #'ror': extension of default implementation which is gray scale androtation invariant
    #'uniform': improved rotation invariance with uniform patterns andfiner quantization of the angular space which is gray scale and rotation invariant.
    #'nri_uniform': non rotation-invariant uniform patterns variantwhich is only gray scale invariant
    #'var': rotation invariant variance measures of the contrast of localimage texture which is rotation but not gray scale invariant
    lbp = skif.local_binary_pattern(image, lbp_point, lbp_radius, 'default')
    # 统计图像的直方图
    max_bins = int(lbp.max() + 1)
    #print(max_bins)
    # hist size:256
    hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return hist

image_path1 = 'lane1.jpg'  #读取图片
# image_path2 = 'lane2.jpg'  #读取图片
image_path3 = 'img.JPG'
feature1 = get_lbp_data(image_path1)  #调用函数
feature2 = get_lbp_data(image_path3)  #调用函数

print(feature1)
print(feature2)


# 绘制直方图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(feature1)
plt.title('LBP Histogram - lane1.jpg')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.plot(feature2)
plt.title('LBP Histogram - lane2.jpg')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 比较特征差异
chi_square = np.sum((feature1 - feature2) ** 2 / (feature1 + feature2))

print('卡方统计量:', chi_square)

