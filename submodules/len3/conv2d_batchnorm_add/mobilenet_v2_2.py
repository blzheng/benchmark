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
        self.conv2d17 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x48, x42):
        x49=self.conv2d17(x48)
        x50=self.batchnorm2d17(x49)
        x51=operator.add(x42, x50)
        return x51

m = M().eval()
x48 = torch.randn(torch.Size([1, 192, 28, 28]))
x42 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x48, x42)
end = time.time()
print(end-start)
