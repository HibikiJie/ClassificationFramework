from torch.utils.data import DataLoader
from utils.lossfunction import FocalLossManyClassification
from utils.dataset import MyDataSet
from sklearn import metrics
from myselfset import MyselfSet
import torch
from torch import nn
import os
import numpy
import warnings
warnings.filterwarnings("ignore")

class Trainer(object):

    def __init__(self):
        self.my_set = MyselfSet()  # 加载设置

        '''查看是否能调用cuda'''
        self.device = self.my_set.device
        print('正在使用%s进行训练' % self.device)

        '''实例化网络,并加载模型参数'''
        self.net = self.my_set.net.to(self.device)

        """加载模型权重"""
        if os.path.exists(f'{self.my_set.net_weights}/last.pt') and self.my_set.is_continue_training:
            self.net.load_state_dict(torch.load(f'{self.my_set.net_weights}/last.pt'))
            print('模型已加载权重参数')
        else:
            print('模型已初始化权重参数')

        '''实例化数据加载集'''
        self.data_set = MyDataSet(self.my_set.data)
        if self.my_set.batch_size >= len(self.data_set):
            drop_last = True
        else:
            drop_last = False
        self.data_loader = DataLoader(self.data_set, self.my_set.batch_size, True, drop_last=drop_last, num_workers=self.my_set.num_workers)
        self.data_set_test = MyDataSet(self.my_set.data, mode='test')
        self.data_loader_test = DataLoader(self.data_set_test, self.my_set.batch_size, True, num_workers=self.my_set.num_workers)

        '''创建损失函数'''
        if self.my_set.is_balance_class:
            alpha = []
            for i in range(self.my_set.num_class):
                alpha.append(self.data_set.alpha[str(i)])
            alpha = numpy.array(alpha)
            alpha = sum(alpha)/alpha
            self.loss_func = FocalLossManyClassification(self.my_set.num_class,gamma=self.my_set.gamma,alpha=alpha,smooth=self.my_set.smooth).to(self.device) # 训练偏移量的损失函数
        else:
            self.loss_func = FocalLossManyClassification(self.my_set.num_class,gamma=self.my_set.gamma,smooth=self.my_set.smooth).to(self.device) # 训练偏移量的损失函数

        '''创建优化器'''
        self.optimizer = torch.optim.Adam(self.net.parameters(),self.my_set.learning_rate)
        if os.path.exists(f'{self.my_set.net_weights}/optimizer.pt'):
            self.optimizer.load_state_dict(torch.load(f'{self.my_set.net_weights}/optimizer.pt'))

        self.min_loss = 1e10
        print('训练器初始化完成')

    def train(self):
        count = 0
        print('训练开始')
        infor = None
        for epoch in range(self.my_set.epoch):
            loss_sum = 0
            '训练'
            self.net.train()
            for images, targets in self.data_loader:

                images = images.to(self.device)
                targets = targets.to(self.device)

                out = self.net(images)
                '''计算损失'''
                loss = self.loss_func(out, targets)
                '''反向传播，梯度更新'''
                loss.backward()
                if (count + 1) % self.my_set.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                loss_sum += loss.item()
                count += 1

            '''验证'''
            self.net.eval()
            c_sum = 0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for images, targets in self.data_loader_test:
                    image = images.to(self.device)
                    targets = targets.to(self.device)
                    out = self.net(image)
                    y_true.extend(targets.cpu().numpy().tolist())
                    out = torch.argmax(out, dim=1)
                    y_pred.extend(out.detach().cpu().numpy().tolist())
                    c = out == targets
                    c = c.sum()
                    c_sum += c
            infor = metrics.classification_report(y_true=y_true,y_pred=y_pred,target_names=self.my_set.class_names)
            logs = f'{epoch}、loss: {loss_sum / len(self.data_loader)} 测试集正确率：{str((c_sum / len(self.data_set_test)).item()*100)[:6]}%'
            print(logs)
            with open(f'{self.my_set.logs}/logs.txt', 'a') as file:
                file.write(logs)
            with open(f'{self.my_set.logs}/classification_report.txt', 'w') as file:
                file.write(str(infor))
            """保存模型参数"""
            self.save_weight(loss_sum / len(self.data_loader))
        print(infor)

    def save_weight(self, loss):
        '''保存模型参数'''
        if loss < self.min_loss:
            self.min_loss = loss
            torch.save(self.net.state_dict(), f'{self.my_set.net_weights}/best.pt')
            torch.save(self.optimizer.state_dict(), f'{self.my_set.net_weights}/optimizer.pt')
        else:
            torch.save(self.net.state_dict(), f'{self.my_set.net_weights}/last.pt')
            torch.save(self.optimizer.state_dict(), f'{self.my_set.net_weights}/optimizer.pt')

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
