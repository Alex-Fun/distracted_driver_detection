import os
import zipfile

dir = "D:\\tmp\data\state-farm-distracted-driver-detection"
img_zip_dir = os.path.join(dir, 'imgs.zip')
f = zipfile.ZipFile(img_zip_dir,'r')
for file in f.namelist():
    f.extract(file, "/output/img/")