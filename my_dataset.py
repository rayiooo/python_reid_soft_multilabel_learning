import os
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir, mode=None, transform=None):
        self.data = [
            x for x in os.scandir(root_dir)
            if x.name.endswith('.jpg')
        ]
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file = self.data[index]
        img = Image.open(file.path, 'r')
        target = int(file.name.split('_')[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, target
