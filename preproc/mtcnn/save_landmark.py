#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
from src import detect_faces, show_bboxes
from PIL import Image


root_path = '../../dataset/celeba/photo'
#root_path = '/home/ludan/code/github/FSGAN/FSGAN/dataset/celeba/test'
image_paths = sorted([os.path.join(root_path, p) for p in os.listdir(root_path)])


# In[ ]:


import numpy as np
from scipy.io import savemat
os.makedirs(
    '/'.join(image_paths[0].replace('\\', '/').replace('/photo/', '/landmark/').split('/')[:-1]),
    exist_ok=True
)
for idx_image_path, image_path in enumerate(image_paths):
    print('{} / {}'.format(idx_image_path+1, len(image_paths)), image_path)
    img = Image.open(image_path).resize((512, 512), Image.BICUBIC)
    bounding_boxes, landmarks = detect_faces(img, min_face_size=10.0)
    show_bboxes(img, bounding_boxes, landmarks)
    if len(landmarks) == 0:
        # 将图片移动到removed文件夹
        os.rename(image_path, image_path.replace('/photo/', '/removed_photo/'))
        print('No face detected, move '+ image_path +' to removed_photo')
        continue
    landmarks = landmarks[0]
    facial5point = []
    for idx_ld in range(5):
        facial5point.append([landmarks[idx_ld], landmarks[idx_ld+5]])
    # 检查是否有landmark超出图片范围（为了FSGAN后期处理）
    x1=facial5point[0][0]
    y1=facial5point[0][1]
    x2=facial5point[1][0]
    y2=facial5point[1][1]
    x3=facial5point[2][0]
    y3=facial5point[2][1]
    x4=facial5point[3][0]
    y4=facial5point[3][1]
    x5=facial5point[4][0]
    y5=facial5point[4][1]
    isNeedSave = (x1>56.0) and (x1<456.0) and (y1>48.0) and (y1<480.0) and (x2>56.0) and (x2<456.0) and (y2>48) and (y2<480.0) and (x3>48.0) and (x3<464.0) and (y3>64.0) and (y3<480.0) and ((x4+x5)>128.0) and ((x4+x5)<896) and ((y4+y5)>80.0) and ((y4+y5)<944)
    if not isNeedSave:
        # 将图片移动到removed文件夹
        os.rename(image_path, image_path.replace('/photo/', '/removed_photo/'))
        print('detect wrong, move '+ image_path +' to removed_photo')
        continue
    np.savetxt(image_path.replace(image_path[-4:], '.txt').replace('\\', '/').replace('/photo/', '/landmark/'), facial5point)

