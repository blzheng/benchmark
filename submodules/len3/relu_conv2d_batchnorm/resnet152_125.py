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
        self.relu124 = ReLU(inplace=True)
        self.conv2d130 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d130 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x429):
        x430=self.relu124(x429)
        x431=self.conv2d130(x430)
        x432=self.batchnorm2d130(x431)
        return x432

m = M().eval()
x429 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x429)
end = time.time()
print(end-start)
