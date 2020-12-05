"""Code to read train and test files and create csv files"""
import pandas as pd
import json
import mat73
import csv

img_type = ['train', 'test']

for type in img_type:

    data_dict = mat73.loadmat(f'../images/{type}/digitStruct.mat')

    with open(f'{type}.csv','w') as f1:

        writer = csv.writer(f1, delimiter=',',lineterminator='\n',)
        writer.writerow(['image','bbox'])

        for idx in range(len(data_dict['digitStruct']['name'])):

            name = data_dict['digitStruct']['name'][idx]
            data = data_dict['digitStruct']['bbox'][idx]

            if isinstance(data['height'], list):
                d = {"height": [int(x) for x in data['height']], "label": [int(x) for x in data['label']], "left": [int(x) for x in data['left']], 'top': [int(x) for x in data['top']], 'width': [int(x) for x in data['width']]}
            else:
                d = {'height': [int(data['height'])], 'label': [int(data['label'])],
                     'left': [int(data['left'])], 'top': [int(data['top'])],
                     'width': [int(data['width'])]}

            row = [name, d]

            writer.writerow(row)

    # Process data into flat format
    # current structure is: image_name, bbox
    # new structure is: image_name, label, up_lft_pt, lwr_rt_pt

    df = pd.DataFrame(columns = ['image_name', 'label', 'up_lft_pt', 'h', 'w'])

    all_data = pd.read_csv(f'../images/{type}/{type}.csv')

    for index, row in all_data.iterrows():
        img_name = row['image']
        data = row['bbox']
        data = data.replace("\'", "\"")
        data = json.loads(data)
        for numb_idx in range(len(data['label'])):
            df = df.append({'image_name': img_name, 'label': data['label'][numb_idx], 'up_lft_pt': tuple((int(data['left'][numb_idx]), int(data['top'][numb_idx]))), 'h': data['height'][numb_idx], 'w': data['width'][numb_idx]}, ignore_index=True)

    df.to_csv(f'../images/{type}/{type}_all.csv')



# Test visualizing to bounding box on original image
# all_data = pd.read_csv('./test_all.csv')
# for index, row in all_data.iterrows():
#     img = row['image_name']
#     print(img)
#     img = cv2.imread(f'./images/test/{img}',0)
#     up_lft_pt = (eval(row['up_lft_pt']))
#     lr_rt_pt = (up_lft_pt[0]-1+row['w'], up_lft_pt[1]-1+ row['h'])
#     cv2.rectangle(img, up_lft_pt, lr_rt_pt, (255, 0, 0), 2)
#     cv2.imshow("lalala", img)
#     k = cv2.waitKey(0)  # 0==wait forever


import numpy as np
from PIL import Image
# img = cv2.imread(f'./images/train/7128.png', cv2.IMREAD_COLOR)
# up_lft_pt = (79, 14) #248,167
# w=34 #31
# h=34 #64
# lr_rt_pt = (up_lft_pt[0]+1+w, up_lft_pt[1]+1+ h)
# img2 = img[int(up_lft_pt[1])+1:int(up_lft_pt[1])+1 + h, int(up_lft_pt[0])+1:int(up_lft_pt[0])+1 + w]
# # cv2.rectangle(img, up_lft_pt, lr_rt_pt, (255, 255, 0), 1)
# cv2.imshow("lalala", img2)
# k = cv2.waitKey(0)