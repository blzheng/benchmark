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
        self.conv2d168 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d100 = BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x501):
        x502=self.conv2d168(x501)
        x503=self.batchnorm2d100(x502)
        return x503

m = M().eval()
x501 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x501)
end = time.time()
print(end-start)
