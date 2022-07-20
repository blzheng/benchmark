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
        self.conv2d100 = Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x358):
        x359=self.conv2d100(x358)
        x360=self.batchnorm2d101(x359)
        return x360

m = M().eval()
x358 = torch.randn(torch.Size([1, 704, 7, 7]))
start = time.time()
output = m(x358)
end = time.time()
print(end-start)
