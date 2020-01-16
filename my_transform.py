from transforms.random_erasing import RandomErasing
from torchvision import transforms


def data_transforms(mode='train', size=(384, 128)):
    assert mode in ['train', 'val', 'test']
    
    mean = [0.485, 0.406, 0.456]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size),
            transforms.RandomCrop(size, 7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomErasing(), ])
    if mode in ['val', 'test']:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std), ])
