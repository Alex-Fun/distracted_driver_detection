import os
import zipfile
import matplotlib.pyplot as plt

base_dir = "/data/oHongMenYan/distracted-driver-detection-dataset"
img_zip_dir = os.path.join(base_dir, 'imgs.zip')
f = zipfile.ZipFile(img_zip_dir,'r')
print('begin')
for file in f.namelist():
    print(file)
    f.extract(file, "/output/img/")
print('done')

# cv2.imwrite(os.path.join(out_dir, title+'.jpg'), cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
