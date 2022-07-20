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
        self.conv2d113 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x329, x334):
        x335=operator.mul(x329, x334)
        x336=self.conv2d113(x335)
        x337=self.batchnorm2d67(x336)
        return x337

m = M().eval()
x329 = torch.randn(torch.Size([1, 1392, 7, 7]))
x334 = torch.randn(torch.Size([1, 1392, 1, 1]))
start = time.time()
output = m(x329, x334)
end = time.time()
print(end-start)
