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
        self.conv2d212 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid37 = Sigmoid()
        self.conv2d213 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d137 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x679, x676):
        x680=self.conv2d212(x679)
        x681=self.sigmoid37(x680)
        x682=operator.mul(x681, x676)
        x683=self.conv2d213(x682)
        x684=self.batchnorm2d137(x683)
        return x684

m = M().eval()
x679 = torch.randn(torch.Size([1, 76, 1, 1]))
x676 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x679, x676)
end = time.time()
print(end-start)
