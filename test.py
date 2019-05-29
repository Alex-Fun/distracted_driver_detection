import os
import zipfile

dir = "D:\\tmp\data\state-farm-distracted-driver-detection"
img_zip_dir = os.path.join(dir, 'imgs.zip')
f = zipfile.ZipFile(img_zip_dir,'r')
print('begin')
for file in f.namelist():
    print(file)
    f.extract(file, "/output/img/")
print('done')
