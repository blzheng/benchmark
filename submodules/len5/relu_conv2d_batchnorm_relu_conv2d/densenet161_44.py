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
        self.relu91 = ReLU(inplace=True)
        self.conv2d91 = Conv2d(1632, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d92 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu92 = ReLU(inplace=True)
        self.conv2d92 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x324):
        x325=self.relu91(x324)
        x326=self.conv2d91(x325)
        x327=self.batchnorm2d92(x326)
        x328=self.relu92(x327)
        x329=self.conv2d92(x328)
        return x329

m = M().eval()
x324 = torch.randn(torch.Size([1, 1632, 14, 14]))
start = time.time()
output = m(x324)
end = time.time()
print(end-start)
