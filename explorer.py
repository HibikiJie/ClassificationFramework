import torch
from models.resnet18 import Net
from myselfset import MyselfSet
import numpy
import os
import cv2


class Explorer:

    def __init__(self, is_cuda=False):
        my_set = MyselfSet()
        self.device = my_set.device
        self.net = Net(my_set.num_class)
        if os.path.exists(f'{my_set.net_weights}/{my_set.load_weight}.pt'):
            self.net.load_state_dict(torch.load(f'{my_set.net_weights}/{my_set.load_weight}.pt', map_location='cpu'))
        else:
            raise RuntimeError('Model parameters are not loaded')
        self.net = self.net.to(self.device).eval()
        self.cls = my_set.class_names
        self.image_size = my_set.image_size

    def __call__(self, image):
        with torch.no_grad():
            h1, w1, _ = image.shape
            max_len = max(h1, w1)
            fx = self.image_size / max_len
            fy = self.image_size / max_len
            image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            h2, w2, _ = image.shape
            background = numpy.zeros((self.image_size, self.image_size, 3), dtype=numpy.uint8)
            s_h = self.image_size//2 - h2 // 2
            s_w = self.image_size//2 - w2 // 2
            background[s_h:s_h + h2, s_w:s_w + w2] = image
            image = background

            image = self.to_tensor(image).unsqueeze(0).to(self.device)
            out = self.net(image).detach().cpu()
            out = torch.argmax(out)

            return self.cls[out.item()]

    @staticmethod
    def to_tensor(image_numpy):
        return torch.from_numpy(image_numpy).float().permute(2, 0, 1) / 255


if __name__ == '__main__':
    import os
    explorer = Explorer(False)
    root = 'data/test'
    for target in os.listdir(root):
        for image_name in os.listdir(f'{root}/{target}'):
            image_path = f'{root}/{target}/{image_name}'
            image = cv2.imread(image_path)
            predict = explorer(image)
            print('预测值：',predict,'  实际值：',explorer.cls[int(target)])
            cv2.imshow('a', image)
            cv2.waitKey()
    cv2.destroyAllWindows()
