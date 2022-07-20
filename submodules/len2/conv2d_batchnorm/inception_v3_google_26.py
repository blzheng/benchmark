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
        self.conv2d26 = Conv2d(288, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.batchnorm2d26 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x99):
        x100=self.conv2d26(x99)
        x101=self.batchnorm2d26(x100)
        return x101

m = M().eval()
x99 = torch.randn(torch.Size([1, 288, 25, 25]))
start = time.time()
output = m(x99)
end = time.time()
print(end-start)
