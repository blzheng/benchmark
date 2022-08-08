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
        self.conv2d186 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d110 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d187 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d111 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x555, x551, x545):
        x556=x555.sigmoid()
        x557=operator.mul(x551, x556)
        x558=self.conv2d186(x557)
        x559=self.batchnorm2d110(x558)
        x560=operator.add(x559, x545)
        x561=self.conv2d187(x560)
        x562=self.batchnorm2d111(x561)
        return x562

m = M().eval()
x555 = torch.randn(torch.Size([1, 1344, 1, 1]))
x551 = torch.randn(torch.Size([1, 1344, 14, 14]))
x545 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x555, x551, x545)
end = time.time()
print(end-start)
