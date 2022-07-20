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
        self.conv2d114 = Conv2d(800, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d115 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x407):
        x408=self.conv2d114(x407)
        x409=self.batchnorm2d115(x408)
        return x409

m = M().eval()
x407 = torch.randn(torch.Size([1, 800, 7, 7]))
start = time.time()
output = m(x407)
end = time.time()
print(end-start)
