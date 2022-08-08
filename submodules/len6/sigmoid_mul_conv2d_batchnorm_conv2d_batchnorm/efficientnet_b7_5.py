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
        self.sigmoid38 = Sigmoid()
        self.conv2d191 = Conv2d(1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d113 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d192 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d114 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x599, x595):
        x600=self.sigmoid38(x599)
        x601=operator.mul(x600, x595)
        x602=self.conv2d191(x601)
        x603=self.batchnorm2d113(x602)
        x604=self.conv2d192(x603)
        x605=self.batchnorm2d114(x604)
        return x605

m = M().eval()
x599 = torch.randn(torch.Size([1, 1344, 1, 1]))
x595 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x599, x595)
end = time.time()
print(end-start)
