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
        self.batchnorm2d37 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d64 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x188):
        x189=self.batchnorm2d37(x188)
        x190=self.conv2d64(x189)
        return x190

m = M().eval()
x188 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x188)
end = time.time()
print(end-start)
