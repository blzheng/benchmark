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
        self.conv2d82 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x256, x241):
        x257=operator.add(x256, x241)
        x258=self.conv2d82(x257)
        x259=self.batchnorm2d48(x258)
        return x259

m = M().eval()
x256 = torch.randn(torch.Size([1, 80, 28, 28]))
x241 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x256, x241)
end = time.time()
print(end-start)
