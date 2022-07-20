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
        self.conv2d47 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x144):
        x145=self.conv2d47(x144)
        x146=self.batchnorm2d27(x145)
        return x146

m = M().eval()
x144 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x144)
end = time.time()
print(end-start)
