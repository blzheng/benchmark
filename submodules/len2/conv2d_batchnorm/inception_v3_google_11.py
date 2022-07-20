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
        self.conv2d11 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x49):
        x50=self.conv2d11(x49)
        x51=self.batchnorm2d11(x50)
        return x51

m = M().eval()
x49 = torch.randn(torch.Size([1, 192, 25, 25]))
start = time.time()
output = m(x49)
end = time.time()
print(end-start)
