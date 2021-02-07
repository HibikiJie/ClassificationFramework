from torch.utils.data import Dataset
from myselfset import MyselfSet
import torch
import numpy
import random
import os
import cv2

class MyDataSet(Dataset):

    def __init__(self, root='data', mode='train'):
        super(MyDataSet, self).__init__()
        self.dataset =[]
        self.my_set = MyselfSet()
        self.alpha = {}
        for target in os.listdir(f'{root}/{mode}'):
            self.alpha[target] = 0
            for file_name in os.listdir(f'{root}/{mode}/{target}'):
                image_name = f'{root}/{mode}/{target}/{file_name}'
                self.dataset.append([image_name,target])
                self.alpha[target] +=1
        self.image_size = self.my_set.image_size
        if len(self.alpha) != self.my_set.num_class:
            raise ValueError('请检查数据样本和设定是否匹配')
        if mode == 'train':
            self.is_train = True
        else:
            self.is_train = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, target = self.dataset[item]
        image = cv2.imread(image_path)
        if self.is_train:
            if random.random()>self.my_set.random_cutting_probability and self.my_set.random_cutting:
                image = self.random_cutting(image)
        image = self.reset_image(image)  # 图片重整为方形
        if self.is_train:
            if random.random()<self.my_set.randomly_adjust_brightness_probability and self.my_set.randomly_adjust_brightness:
                image = self.randomly_adjust_brightness(image,random.randint(-50,50),random.randint(-50,50))

            if random.random()<self.my_set.salt_noise_probability and self.my_set.randomly_adjust_brightness:
                image = self.sp_noise(image,self.my_set.salt_noise_probs)

            if random.random()<self.my_set.to_hsv_probs and self.my_set.to_hsv:
                if random.random()>0.5:
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # cv2.imshow('a',image)
        # cv2.waitKey()
        image = torch.from_numpy(image).float().permute(2,0,1)/255
        target = torch.tensor(int(target)).long()
        return image,target

    @staticmethod
    def sp_noise(image, prob):
        '''
        添加椒盐噪声
        prob:噪声比例
        '''
        output = numpy.zeros(image.shape, numpy.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = random.randint(0, 255)
                elif rdn > thres:
                    output[i][j] = random.randint(0, 255)
                else:
                    output[i][j] = image[i][j]
        return output

    @staticmethod
    def randomly_adjust_brightness(image, lightness, saturation):
        # 颜色空间转换 BGR转为HLS
        image = image.astype(numpy.float32) / 255.0
        hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        # 1.调整亮度（线性变换)
        hlsImg[:, :, 1] = (1.0 + lightness / float(100)) * hlsImg[:, :, 1]
        hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
        # 饱和度
        hlsImg[:, :, 2] = (1.0 + saturation / float(100)) * hlsImg[:, :, 2]
        hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
        # HLS2BGR
        lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
        lsImg = lsImg.astype(numpy.uint8)
        return lsImg

    def reset_image(self,image):

        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        fx = self.image_size / max_len
        fy = self.image_size / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, _ = image.shape
        background = numpy.zeros((self.image_size, self.image_size, 3), dtype=numpy.uint8)
        if self.is_train:
            s_h = random.randint(0,self.image_size-h2)
            s_w = random.randint(0,self.image_size-w2)
        else:
            s_h = self.image_size // 2 - h2 // 2
            s_w = self.image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        return background

    def random_cutting(self,image):
        h,w,_ = image.shape
        a = random.randint(0,self.my_set.random_cutting_size)
        b = random.randint(0, self.my_set.random_cutting_size)
        image = image[a:h-a,b:w-b]
        return image


if __name__ == '__main__':
    a = MyDataSet('/home/cq/pubilic/hibiki/ClassificationNetwork/data',mode='train')
    for i in range(10000):
        print(a[2])
