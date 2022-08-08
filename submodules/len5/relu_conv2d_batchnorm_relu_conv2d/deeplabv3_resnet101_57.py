import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.relu85 = ReLU(inplace=True)
        self.conv2d91 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d91 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu88 = ReLU(inplace=True)
        self.conv2d92 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)

    def forward(self, x301):
        x302=self.relu85(x301)
        x303=self.conv2d91(x302)
        x304=self.batchnorm2d91(x303)
        x305=self.relu88(x304)
        x306=self.conv2d92(x305)
        return x306

m = M().eval()
x301 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x301)
end = time.time()
print(end-start)
