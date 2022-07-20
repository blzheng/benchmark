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
        self.conv2d7 = Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False)
        self.batchnorm2d7 = BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d8 = Conv2d(88, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x34):
        x35=self.conv2d7(x34)
        x36=self.batchnorm2d7(x35)
        x37=self.conv2d8(x36)
        return x37

m = M().eval()
x34 = torch.randn(torch.Size([1, 88, 28, 28]))
start = time.time()
output = m(x34)
end = time.time()
print(end-start)
