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
        self.conv2d62 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d63 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x186, x174):
        x187=self.conv2d62(x186)
        x188=self.batchnorm2d36(x187)
        x189=operator.add(x188, x174)
        x190=self.conv2d63(x189)
        return x190

m = M().eval()
x186 = torch.randn(torch.Size([1, 384, 28, 28]))
x174 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x186, x174)
end = time.time()
print(end-start)
