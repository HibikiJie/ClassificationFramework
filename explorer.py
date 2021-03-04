import torch
from utils import enhance
from models.resnet18 import Net
from myselfset import MyselfSet
import numpy
import os
import cv2


class Explorer:

    def __init__(self, is_cuda=False):
        """

        :param is_cuda: 是否使用gpu
        """
        my_set = MyselfSet()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else "cpu")
        self.net = my_set.net
        """加载网络权重"""
        if os.path.exists(f'{my_set.net_weights}/{my_set.load_weight}.pt'):
            self.net.load_state_dict(torch.load(f'{my_set.net_weights}/{my_set.load_weight}.pt', map_location='cpu'))
        else:
            raise RuntimeError('模型参数未能加载成功')
        self.net = self.net.to(self.device).eval()
        self.cls = my_set.class_names
        self.image_size = my_set.image_size

    def __call__(self, image):
        with torch.no_grad():
            image = enhance.square_picture(image, self.image_size)  # 图片正方形化
            image = self.to_tensor(image).unsqueeze(0).to(self.device)  # 转为tensor，并归一化
            out = self.net(image).detach().cpu()  # 预测输出结果
            out = torch.argmax(out)

            return self.cls[out.item()]

    @staticmethod
    def to_tensor(image_numpy):
        return torch.from_numpy(image_numpy).float().permute(2, 0, 1) / 255


if __name__ == '__main__':
    import os

    explorer = Explorer(False)
    root = 'data/test'
    count = 0
    num_correct = 0
    for target in os.listdir(root):
        for image_name in os.listdir(f'{root}/{target}'):
            image_path = f'{root}/{target}/{image_name}'
            image = cv2.imread(image_path)
            predict = explorer(image)
            print('预测值：', predict, '  实际值：', explorer.cls[int(target)])
            if predict == explorer.cls[int(target)]:
                num_correct += 1
            count += 1
            cv2.imshow('a', image)
            cv2.waitKey(1)
    print("正确率：", num_correct / count)
    cv2.destroyAllWindows()
