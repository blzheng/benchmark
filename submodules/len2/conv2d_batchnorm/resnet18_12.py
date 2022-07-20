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
        self.conv2d12 = Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d12 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x34):
        x40=self.conv2d12(x34)
        x41=self.batchnorm2d12(x40)
        return x41

m = M().eval()
x34 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x34)
end = time.time()
print(end-start)
