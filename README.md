此图像分类库，可快速完成图像分类任务。鉴于有些场景需要分类目标旋向问题，故并未使用旋转之类的数据增强手段。所有代码和模型都在积极开发中，可以修改或删除，恕不另行通知。

# 要求

Python 3.8 或更晚，安装requirements.txt依赖项，包括 。要安装运行：`torch>=1.7`

```
$ pip install -r requirements.txt
```



# 教程

所有设置项均在[myselfset.py](myselfset.py)中.

数据的制作：

在data文件中，分别按照文件夹的新式对图片进行分类，即可开始训练。



# 推理

explorer.py演示了对data/test中的图片进行分类推理的过程。



# 训练

配置好myselfset.py的文件中的设置项后，主要是类别名称的配置。运行train.py文件即可开始训练

```
$ python train.py
```

