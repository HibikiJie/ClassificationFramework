from torchvision.models import resnet18
from torch import nn
import torch

class Net(nn.Module):

    def __init__(self,out_features):
        super(Net, self).__init__()
        self.headbone = resnet18(True)
        self.headbone.fc = nn.Sequential(
            nn.Linear(512, out_features)
        )

    def forward(self,x):
        x = self.headbone(x)
        return x

if __name__ == '__main__':
    m = Net(3)
    x = torch.randn(2,3,112,112)
    print(m(x).shape)