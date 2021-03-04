from torch.utils.data import DataLoader
from utils.lossfunction import FocalLossManyClassification
from utils.dataset import MyDataSet
from sklearn import metrics
from myselfset import MyselfSet
import torch
import os
import numpy
import warnings

warnings.filterwarnings("ignore")  # 忽略警告


class Trainer(object):

    def __init__(self):
        self.my_set = MyselfSet()  # 加载设置

        '''查看是否能调用cuda'''
        self.device = self.my_set.device
        print('准备使用%s设备进行训练' % self.device)

        '''实例化网络,并加载模型参数'''
        self.net = self.my_set.net.to(self.device)
        print(self.net)
        """加载模型权重"""
        if os.path.exists(f'{self.my_set.net_weights}/last.pt') and self.my_set.is_continue_training:
            self.net.load_state_dict(torch.load(f'{self.my_set.net_weights}/last.pt'))
            print('模型已加载权重参数')
        else:
            print('模型已初始化权重参数')

        '''实例化训练数据集'''
        self.data_set = MyDataSet(self.my_set.data)
        if self.my_set.batch_size >= len(self.data_set):
            drop_last = True  # 抛弃最后一个批次，数据集大小可能不能被batch size整除，故最后一个批次会小于batch size，可抛弃
        else:
            drop_last = False

        """实例化训练数据加载器"""
        self.data_loader = DataLoader(self.data_set, self.my_set.batch_size, True, drop_last=drop_last,
                                      num_workers=self.my_set.num_workers)

        """实例化测试集，和测试数据加载器"""
        self.data_set_test = MyDataSet(self.my_set.data, mode='test')
        self.data_loader_test = DataLoader(self.data_set_test, self.my_set.batch_size, True,
                                           num_workers=self.my_set.num_workers)

        '''创建损失函数'''
        if self.my_set.is_balance_class:  # 如果平衡类别
            """计算平衡系数"""
            alpha = []
            for i in range(self.my_set.num_class):
                alpha.append(self.data_set.alpha[str(i)])
            alpha = numpy.array(alpha)
            alpha = sum(alpha) / alpha
            """创建带类别数量平衡的损失函数"""
            self.loss_func = FocalLossManyClassification(self.my_set.num_class, gamma=self.my_set.gamma, alpha=alpha,
                                                         smooth=self.my_set.smooth).to(self.device)
        else:  # 不平衡类别数量差异
            """创建一般损失函数"""
            self.loss_func = FocalLossManyClassification(self.my_set.num_class, gamma=self.my_set.gamma,
                                                         smooth=self.my_set.smooth).to(self.device)

        '''创建优化器'''
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.my_set.learning_rate)
        if os.path.exists(f'{self.my_set.net_weights}/optimizer.pt') and self.my_set.is_continue_training:
            self.optimizer.load_state_dict(torch.load(f'{self.my_set.net_weights}/optimizer.pt'))

        self.correct = 0
        print('训练器初始化完成')

    def train(self):
        count = 0
        print('训练开始')
        infor = None
        for epoch in range(self.my_set.epoch):
            loss_sum = 0
            '训练'
            self.net.train()  # 开启训练模式
            for images, targets in self.data_loader:

                images = images.to(self.device)
                targets = targets.to(self.device)

                y = self.net(images)  # 预测

                '''计算损失'''
                loss = self.loss_func(y, targets)

                '''反向传播，梯度更新'''
                loss.backward()
                if (count + 1) % self.my_set.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                """统计信息"""
                loss_sum += loss.item()
                count += 1

            '''验证'''
            self.net.eval()  # 开启使用模式
            c_sum = 0
            y_true = []
            y_pred = []
            with torch.no_grad():  # 关闭梯度追踪
                for images, targets in self.data_loader_test:
                    image = images.to(self.device)
                    targets = targets.to(self.device)
                    out = self.net(image)  # 预测
                    y_true.extend(targets.cpu().numpy().tolist())
                    out = torch.argmax(out, dim=1)
                    y_pred.extend(out.detach().cpu().numpy().tolist())
                    c = out == targets  # 判断预测值和标签是否相等
                    c = c.sum()  # 计算预测正确的个数
                    c_sum += c
            """统计信息"""
            infor = metrics.classification_report(y_true=y_true, y_pred=y_pred, target_names=self.my_set.class_names)
            logs = f'{epoch}、loss: {loss_sum / len(self.data_loader)} 测试集正确率：{str((c_sum / len(self.data_set_test)).item() * 100)[:6]}%'
            print(logs)
            with open(f'{self.my_set.logs}/logs.txt', 'a') as file:
                file.write(logs + '\n')
            with open(f'{self.my_set.logs}/classification_report.txt', 'w') as file:
                file.write(str(infor))
            """保存模型参数"""
            self.save_weight((c_sum / len(self.data_set_test)).item())

        print('训练完成')
        print(infor)

    def save_weight(self, correct):
        '''保存模型参数'''
        # print(correct, self.correct)
        if correct >= self.correct:
            self.correct = correct
            torch.save(self.net.state_dict(), f'{self.my_set.net_weights}/best.pt')
            torch.save(self.optimizer.state_dict(), f'{self.my_set.net_weights}/optimizer.pt')
        else:
            torch.save(self.net.state_dict(), f'{self.my_set.net_weights}/last.pt')
            torch.save(self.optimizer.state_dict(), f'{self.my_set.net_weights}/optimizer.pt')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
