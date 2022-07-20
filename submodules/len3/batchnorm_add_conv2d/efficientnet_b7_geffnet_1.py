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
        self.batchnorm2d50 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d87 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x260, x247):
        x261=self.batchnorm2d50(x260)
        x262=operator.add(x261, x247)
        x263=self.conv2d87(x262)
        return x263

m = M().eval()
x260 = torch.randn(torch.Size([1, 80, 28, 28]))
x247 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x260, x247)
end = time.time()
print(end-start)
