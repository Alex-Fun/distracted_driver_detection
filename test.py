import os
import zipfile
import matplotlib.pyplot as plt

# base_dir = "/data/oHongMenYan/distracted-driver-detection-dataset"
# img_zip_dir = os.path.join(base_dir, 'imgs.zip')
# f = zipfile.ZipFile(img_zip_dir,'r')
# print('begin')
# for file in f.namelist():
#     print(file)
#     f.extract(file, "/output/img/")
# print('done')

# cv2.imwrite(os.path.join(out_dir, title+'.jpg'), cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

import cv2 as cv
filename = r'D:\Users\Alex\Pictures\IDR_THEME_NTP_BACKGROUND@2x.png'

img = cv.imread(filename)
# 高斯模糊
imgGauss = cv.GaussianBlur(img, (5, 5), 0)
# img.shape=[Height, Width] 以矩阵的方式描述img的shape，而不是日常习惯的[宽, 高]的顺序
image1 = cv.resize(img, dsize=(int(img.shape[1]/2), int(img.shape[0]/2)))
image2 = cv.pyrDown(image1)
# 颜色空间转换
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# 阈值化
_, gray1 = cv.threshold(gray, 120, 0xff, cv.THRESH_BINARY)
plt.figure()
plt.imshow(img)
plt.imshow(rgb_img)
plt.plot()


# cv.imshow("source image", img)
# cv.imshow("rbg_img image", rgb_img)
# cv.imshow("Gaussian filtered image", imgGauss)
# cv.imshow("half size", image1)
# cv.imshow("quarter size", image2)
# cv.imshow("gray", gray)
# cv.imshow("threshold image", gray1)

# cv.waitKey()
# cv.destroyAllWindows()