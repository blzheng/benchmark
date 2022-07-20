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
        self.conv2d157 = Conv2d(1200, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d93 = BatchNorm2d(344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x492):
        x493=self.conv2d157(x492)
        x494=self.batchnorm2d93(x493)
        return x494

m = M().eval()
x492 = torch.randn(torch.Size([1, 1200, 7, 7]))
start = time.time()
output = m(x492)
end = time.time()
print(end-start)
