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
        self.conv2d146 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid29 = Sigmoid()
        self.conv2d147 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d87 = BatchNorm2d(304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x455, x452):
        x456=self.conv2d146(x455)
        x457=self.sigmoid29(x456)
        x458=operator.mul(x457, x452)
        x459=self.conv2d147(x458)
        x460=self.batchnorm2d87(x459)
        return x460

m = M().eval()
x455 = torch.randn(torch.Size([1, 76, 1, 1]))
x452 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x455, x452)
end = time.time()
print(end-start)
