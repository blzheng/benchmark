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
        self.conv2d72 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x216, x202):
        x217=operator.add(x216, x202)
        x218=self.conv2d72(x217)
        x219=self.batchnorm2d42(x218)
        return x219

m = M().eval()
x216 = torch.randn(torch.Size([1, 80, 28, 28]))
x202 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x216, x202)
end = time.time()
print(end-start)
