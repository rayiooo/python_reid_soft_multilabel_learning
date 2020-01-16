import cv2
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get(path, classes=None):
    '''获取numpy图片数组和onehot标签。
    return: images, labels
    '''
    images = []
    labels = []
    
    train_image_path = sorted(list(list_images(path)))
    for image_path in train_image_path:
        # read image
        image = cv2.imread(image_path)
        images.append(image)

        # read label
        label = int(image_path.split(os.path.sep)[-1].split('_')[0])
        labels.append(label)

    images = np.array(images, dtype='float')
    labels = np.array(labels)
    classes = classes or set(labels)
    print('Found %d images belonging to %d classes.' % (len(images), len(set(labels))))

    # OneHot
    labels = labels.reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto')
    encoder.fit([[x] for x in range(classes)])
    labels = encoder.transform(labels).toarray()
    
    return images, labels


def list_images(basePath, contains=None):
    # 返回有效的图片路径数据集
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # 遍历图片数据目录，生成每张图片的路径
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # 循环遍历当前目录中的文件名
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue
 
            # 通过确定.的位置，从而确定当前文件的文件扩展名
            ext = filename[filename.rfind("."):].lower()
 
            # 检查文件是否为图像，是否应进行处理
            if validExts is None or ext.endswith(validExts):
                # 构造图像路径
                imagePath = os.path.join(rootDir, filename)
                yield imagePath
 