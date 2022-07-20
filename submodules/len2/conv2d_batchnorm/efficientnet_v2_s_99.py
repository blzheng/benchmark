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
        self.conv2d153 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d99 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x487):
        x488=self.conv2d153(x487)
        x489=self.batchnorm2d99(x488)
        return x489

m = M().eval()
x487 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x487)
end = time.time()
print(end-start)
