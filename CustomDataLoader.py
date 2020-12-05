import pandas as pd
import cv2
import os

from torch.utils.data import Dataset

class CustomDataLoader(Dataset):
    """Puts the image and label together and resizes"""

    def __init__(self, image_folder, file_path, transform=None):
        datas = pd.read_csv(file_path)
        datas = datas.set_index('Unnamed: 0')
        datas = datas.dropna(axis=0)
        self.image_names = datas.loc[:, 'image_name']
        self.labels = datas.loc[:, 'label']
        self.data = datas.iloc[:, 1:]
        self.transform = transform
        self.image_folder = image_folder

    def __len__(self):
        return self.data.shape[0] - 1

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)

        # read image and crop out number
        image_name = self.image_names[index]
        # print(image_name)
        img_path = os.path.join(self.image_folder, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        up_lft_pt = eval(self.data.loc[index, 'up_lft_pt'])
        w = int(self.data.loc[index, 'w'])
        h = int(self.data.loc[index, 'h'])

        # Extract image
        if int(up_lft_pt[0]) + w - 1 > img.shape[1] or int(up_lft_pt[1]) + h - 1 > img.shape[0]:
            if int(up_lft_pt[0]) + 1 + w > img.shape[1]:  # cols check
                image = img[int(up_lft_pt[1]) + 1:int(up_lft_pt[1]) + 1 + h, int(up_lft_pt[0]) + 1:img.shape[1]]
            else:
                image = img[int(up_lft_pt[1]) + 1:img.shape[0], int(up_lft_pt[0]) + 1:int(up_lft_pt[0]) + 1 + w]
        elif int(up_lft_pt[0]) + 1 < 0:
            image = img[int(up_lft_pt[1]) + 1:int(up_lft_pt[1]) + 1 + h, 0:0 + w]
        elif int(up_lft_pt[1]) + 1 < 0:
            image = img[0:0 + h, int(up_lft_pt[0]) + 1:int(up_lft_pt[0]) + 1 + w]
        else:
            image = img[int(up_lft_pt[1]) + 1:int(up_lft_pt[1]) + 1 + h, int(up_lft_pt[0]) + 1:int(up_lft_pt[0]) + 1 + w]

        # resize upscale
        try:
            image = cv2.resize(image, (224, 224))
        except:
            print(f'failed on image {image_name}')

        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
