import os
import re
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, require_view=False, encode_label=False):
        '''
        :param root_dir: str, tuple or list
        :param transform:
        :param require_view: 是否需要摄像头编号信息
        '''
        if isinstance(root_dir, str):
            self.data = [x for x in os.scandir(root_dir) if x.name.endswith('.jpg')]
        elif isinstance(root_dir, (tuple, list)):
            self.data = []
            for root in root_dir:
                self.data.extend([x for x in os.scandir(root) if x.name.endswith('.jpg')])
        self.data.sort(key=lambda x: x.name)
        
        self.label = [
            int(f.name.split('_')[0]) 
            for f in self.data
        ]
        if encode_label:
            self.label = LabelEncoder().fit_transform(self.label)
        self.view = [
            int(re.search(r'c\d+', file.name).group()[1:])
            for file in self.data
        ]  # 摄像头编号
        
        self.transform = transform
        self.require_view = require_view

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file = self.data[index]
        img = Image.open(file.path, 'r')
        label = self.label[index]
        view = self.view[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.require_view:
            return img, label, view, index
        return img, label
