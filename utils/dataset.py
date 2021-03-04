from torch.utils.data import Dataset
from myselfset import MyselfSet
import torch
import numpy
import random
import os
import cv2
from utils import enhance


class MyDataSet(Dataset):
    """
    数据集
    参数：
        root:数据存放地址。在root文件夹下，分别为各个类别的文件夹，在各个类别文件夹下存放的是该类别的图片。
        mode：初始化该数据集时的模式，只能选择，train和test
    """

    def __init__(self, root='data', mode='train'):
        super(MyDataSet, self).__init__()
        self.dataset = []
        self.my_set = MyselfSet()
        self.alpha = {}
        if mode == 'train':
            self.is_train = True
        elif mode == 'test':
            self.is_train = False
        else:
            raise ValueError('模式错误')

        """加载图片数据路径及其标签"""
        for target in os.listdir(f'{root}/{mode}'):
            self.alpha[target] = 0  # 统计各类别的数量，以便应用于类别平衡
            for file_name in os.listdir(f'{root}/{mode}/{target}'):
                image_name = f'{root}/{mode}/{target}/{file_name}'
                self.dataset.append([image_name, target])
                self.alpha[target] += 1

        self.image_size = self.my_set.image_size
        if len(self.alpha) != self.my_set.num_class:
            raise ValueError('请检查数据样本和设定是否匹配')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """加载数据"""
        image_path, target = self.dataset[item]  # 获取数据（图片路径、标签）
        image = cv2.imread(image_path)  # 加载图片
        image = self.data_to_enhance(image)  # 数据增强
        # cv2.imshow('a', image)
        # cv2.waitKey()
        """转换数据格式为tensor"""
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        target = torch.tensor(int(target)).long()
        # print(image.shape)
        return image, target

    def data_to_enhance(self, image):
        """
        数据增强
            包含图片随机裁剪，随机明亮度变化，随机饱和度变化，随机椒盐噪声，随机色彩变化，高斯模糊，空间扭曲
        :param image: 输入图片
        :return: image: 输出图片
        """
        if self.is_train:
            """图片随机裁剪"""
            if random.random() > self.my_set.random_cutting_probability and self.my_set.random_cutting:
                image = enhance.random_cutting(image, self.my_set.random_cutting_size)

        '''图片调整为正方形'''
        image = enhance.reset_image(image, self.my_set.image_size, self.is_train)  # 图片重整为方形

        if self.is_train:
            """随机明亮饱和度变化"""

            if random.random() < self.my_set.randomly_adjust_brightness_probability and self.my_set.randomly_adjust_brightness:
                image = enhance.randomly_adjust_brightness(image, random.randint(-50, 50), random.randint(-50, 50))

            '''随机椒盐噪声'''
            if random.random() < self.my_set.salt_noise_probability and self.my_set.randomly_adjust_brightness:
                image = enhance.sp_noise(image, self.my_set.salt_noise_probs)

            '''随机hsv色彩变化'''
            if random.random() < self.my_set.to_hsv_probs and self.my_set.to_hsv:
                if random.random() > 0.5:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            """随机高斯模糊"""
            if random.random() < self.my_set.gauss_blur_probs and self.my_set.gauss_blur_probs:
                image = enhance.gauss_blur(image, self.my_set.gauss_blur_max_level)

            """随机空间扭曲"""
            if random.random() < self.my_set.distortion_probs and self.my_set.random_space_distortion:
                image, points = enhance.augment_sample(image, self.image_size * 2)
                image = enhance.reconstruct_image(image, [numpy.array(points).reshape((2, 4))],
                                                  (self.image_size, self.image_size))[0]

        return image


if __name__ == '__main__':
    a = MyDataSet('/media/cq/data/public/hibiki/ClassificationNetwork/data', mode='train')
    for i in range(10000):
        print(a[2])
