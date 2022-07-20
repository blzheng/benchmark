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
        self.conv2d13 = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x48, x42):
        x49=operator.add(x48, x42)
        x50=self.conv2d13(x49)
        x51=self.batchnorm2d13(x50)
        return x51

m = M().eval()
x48 = torch.randn(torch.Size([1, 64, 56, 56]))
x42 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x48, x42)
end = time.time()
print(end-start)
