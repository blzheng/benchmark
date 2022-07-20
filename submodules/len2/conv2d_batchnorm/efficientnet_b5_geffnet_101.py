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
        self.conv2d169 = Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
        self.batchnorm2d101 = BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x504):
        x505=self.conv2d169(x504)
        x506=self.batchnorm2d101(x505)
        return x506

m = M().eval()
x504 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x504)
end = time.time()
print(end-start)
