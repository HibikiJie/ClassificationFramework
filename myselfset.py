import torch
from models.resnet18 import Net
from models.resnet_transformer import BotNet


class MyselfSet:

    def __init__(self):
        self.net = None  # 网络模型，不可实例化，默认resnet18网络
        # self.net = BotNet  # 示例：可额外赋值其他网络进行分类，勿实例化
        self.batch_size = 64  # 训练的批次
        self.device = 'cuda:0'  # 选择训练设备,可选项'cuda device, i.e. 0 or 0,1,2,3 or cpu'
        self.image_size = 112  # 图片尺寸
        # self.image_channel = 3  # 图片通道
        self.epoch = 1000  # 总计训练轮次
        self.num_workers = 4  # 训练中加载数据的核心数
        self.net_weights = 'weights'  # 网络权重保存地址
        self.data = 'data'  # 训练数据地址
        self.is_continue_training = False  # 是否继续上一次的结果训练
        self.accumulation_steps = 1  # 梯度累计的步数,
        self.load_weight = 'best'  # 加载权重的模式，best or last
        self.logs = 'weights'  # 日志文件保存地址
        self.learning_rate = 1e-3  # 学习率
        self.gamma = 2  # 损失gamma参数，平衡简单样本和困难样本学习力度，推荐 2
        self.is_balance_class = False  # 做损失时是否平衡样本权重，True时，自动根据样本数量比值简单平衡，可自行自定义。
        self.smooth = 0.1  # 标签平滑处理参数,可选项'None or range(0,1)'

        """类别名称"""
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']

        """数据增强选项"""
        #  椒盐噪声配置
        self.salt_noise = True  # 是否添加椒盐噪声
        self.salt_noise_probs = 0.02  # 椒盐噪声的比例
        self.salt_noise_probability = 0.5  # 添加椒盐噪声的概率
        #  明亮度饱和度的随机变化配置
        self.randomly_adjust_brightness = True  # 是否开启随机明亮度变化
        self.randomly_adjust_brightness_probability = 0.5  # 随机明亮度变化的概率
        #  随机剪裁配置
        self.random_cutting = True  # 是否随机裁切
        self.random_cutting_probability = 0.5
        # 随机色彩变化配置
        self.to_hsv = True  # 是否开始色彩空间转换增强
        self.to_hsv_probs = 0.5  # 色彩空间增强的概率
        self.random_cutting_size = 5
        # 高斯模糊配置
        self.gauss_blur = True  # 是否开启随机高斯模糊
        self.gauss_blur_probs = 0.5  # 高斯模糊的概率
        self.gauss_blur_max_level = 8  # 高斯的最高模糊等级
        # 随机空间扭曲配置
        self.random_space_distortion = True  # 是否开启随机空间扭曲
        self.distortion_probs = 0.5  # 空间扭曲的概率

        """根据配置计算的参数"""
        self.num_class = len(self.class_names)
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if self.net is None:
            self.net = Net(self.num_class)
        else:
            self.net = self.net(self.num_class)
