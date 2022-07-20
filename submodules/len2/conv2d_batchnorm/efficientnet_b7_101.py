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
        self.conv2d171 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x537):
        x538=self.conv2d171(x537)
        x539=self.batchnorm2d101(x538)
        return x539

m = M().eval()
x537 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x537)
end = time.time()
print(end-start)
