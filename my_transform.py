from transforms.random_erasing import RandomErasing
from torchvision import transforms


def data_transforms(mode='train', size=(256, 128)):
    assert mode in ['train', 'val', 'test']

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(size),
            transforms.Pad(10),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            RandomErasing(), ])
    if mode in ['val', 'train']:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]), ])
