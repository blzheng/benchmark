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
        self.batchnorm2d35 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d60 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x179):
        x180=self.batchnorm2d35(x179)
        x181=self.conv2d60(x180)
        return x181

m = M().eval()
x179 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x179)
end = time.time()
print(end-start)