import cv2
import numpy as np

# 计算LBP等价模式
def compute_lbp(image):
    height, width = image.shape[:2]
    lbp_image = np.zeros_like(image)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = image[y, x]
            code = 0
            powers = [1, 2, 4, 8, 16, 32, 64, 128]
            neighbors = [image[y - 1, x - 1], image[y - 1, x], image[y - 1, x + 1],
                         image[y, x + 1], image[y + 1, x + 1], image[y + 1, x],
                         image[y + 1, x - 1], image[y, x - 1]]

            for i in range(8):
                if neighbors[i] >= center:
                    code += powers[i]

            lbp_image[y, x] = code

    return lbp_image

# 计算LBP等价模式表
def compute_lbp_table():
    table = np.zeros(256, dtype=np.uint8)
    label = 1

    for i in range(256):
        binary = bin(i)[2:].zfill(8)  # 转为8位二进制数
        transitions = binary.count('01') + binary.count('10')  # 计算01或10的转换次数

        if transitions <= 2:
            table[i] = label
            label += 1

    return table

# 加载图像并转换为灰度图像
image = cv2.imread('lane1.jpg', 0)

# 计算LBP等价模式
lbp_image = compute_lbp(image)

# 计算LBP等价模式表
lbp_table = compute_lbp_table()

# 对LBP等价模式进行映射
lbp_image_mapped = lbp_table[lbp_image]

# 显示原始图像和LBP等价模式图像
# cv2.imshow('原始图像', image)
cv2.imshow('LBP等价模式图像', lbp_image_mapped)
cv2.waitKey(0)
cv2.destroyAllWindows()
