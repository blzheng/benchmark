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
        self.conv2d122 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()
        self.conv2d123 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d81 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x388, x385):
        x389=self.conv2d122(x388)
        x390=self.sigmoid20(x389)
        x391=operator.mul(x390, x385)
        x392=self.conv2d123(x391)
        x393=self.batchnorm2d81(x392)
        return x393

m = M().eval()
x388 = torch.randn(torch.Size([1, 64, 1, 1]))
x385 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x388, x385)
end = time.time()
print(end-start)
