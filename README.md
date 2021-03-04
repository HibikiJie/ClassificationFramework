此图像分类库，可快速完成图像分类任务。鉴于有些场景需要分类目标旋向问题，故并未使用旋转之类的数据增强手段。所有代码和模型都在积极开发中，可以修改或删除，恕不另行通知。



# 要求

Python 3.8 或更晚，安装requirements.txt依赖项，包括 。要安装运行：`torch>=1.7`

```
$ pip install -r requirements.txt
```

- [Python3](https://www.python.org/)
  - [Pytorch](https://pytorch.org/)
  - [Numpy](https://www.numpy.org/)
  - [opencv-python](https://github.com/skvark/opencv-python)





# 教程

所有设置项均在[myselfset.py](myselfset.py)中.

数据的制作：

在data文件中，分别按照文件夹的形式对图片进行分类，即可开始训练。



```
python train.py
```

即可直接训练，在data文件夹下的示例图片。

并且可以

```
python explorer.py
```

对测试集进行推理使用





# 环境

可在以下任何最新验证环境中运行（所有依赖项（包括[CUDA](https://developer.nvidia.com/cuda)/ CUDNN、Python和[PyTorch](https://pytorch.org/)预装）：

- **谷歌科拉布和卡格尔**笔记本与免费的GPU：[![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) [![Open In Kaggle](https://camo.githubusercontent.com/a08ca511178e691ace596a95d334f73cf4ce06e83a5c4a5169b8bb68cac27bef/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667)](https://www.kaggle.com/ultralytics/yolov5)
- **谷歌云**深度学习虚拟市场。请参阅[GCP 快速启动指南](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
- **亚马逊**深度学习阿米。请参阅[AWS 快速启动指南](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart)
- **多克图像**.查看[多克快速启动指南](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) [![Docker Pulls](https://camo.githubusercontent.com/280faedaf431e4c0c24fdb30ec00a66d627404e5c4c498210d3f014dd58c2c7e/68747470733a2f2f696d672e736869656c64732e696f2f646f636b65722f70756c6c732f756c7472616c79746963732f796f6c6f76353f6c6f676f3d646f636b6572)](https://hub.docker.com/r/ultralytics/yolov5)





# 推理

explorer.py演示了对data/test中的图片进行分类推理的过程。





# 训练

配置好myselfset.py的文件中的设置项后，主要是类别名称的配置。运行train.py文件即可开始训练

```
$ python train.py
```
训练过程中，会自动保存模型参数于‘weights’文件夹下，并生成目前权重下，在测试中的分类表现报告于‘weights’文件夹。
