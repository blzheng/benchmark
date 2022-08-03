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
        self.batchnorm2d122 = BatchNorm2d(1296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu122 = ReLU(inplace=True)
        self.conv2d122 = Conv2d(1296, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d123 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu123 = ReLU(inplace=True)
        self.conv2d123 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x433):
        x434=self.batchnorm2d122(x433)
        x435=self.relu122(x434)
        x436=self.conv2d122(x435)
        x437=self.batchnorm2d123(x436)
        x438=self.relu123(x437)
        x439=self.conv2d123(x438)
        return x439

m = M().eval()
x433 = torch.randn(torch.Size([1, 1296, 7, 7]))
start = time.time()
output = m(x433)
end = time.time()
print(end-start)
