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
        self.conv2d84 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu79 = ReLU(inplace=True)
        self.conv2d85 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x278, x272):
        x279=self.conv2d84(x278)
        x280=self.batchnorm2d84(x279)
        x281=operator.add(x280, x272)
        x282=self.relu79(x281)
        x283=self.conv2d85(x282)
        x284=self.batchnorm2d85(x283)
        return x284

m = M().eval()
x278 = torch.randn(torch.Size([1, 256, 28, 28]))
x272 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x278, x272)
end = time.time()
print(end-start)
