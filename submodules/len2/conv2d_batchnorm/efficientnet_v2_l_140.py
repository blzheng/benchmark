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
        self.conv2d212 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d140 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x683):
        x684=self.conv2d212(x683)
        x685=self.batchnorm2d140(x684)
        return x685

m = M().eval()
x683 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x683)
end = time.time()
print(end-start)
