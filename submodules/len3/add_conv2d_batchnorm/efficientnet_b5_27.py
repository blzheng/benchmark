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
        self.batchnorm2d100 = BatchNorm2d(1824, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x525, x510):
        x526=operator.add(x525, x510)
        x527=self.conv2d168(x526)
        x528=self.batchnorm2d100(x527)
        return x528

m = M().eval()
x525 = torch.randn(torch.Size([1, 304, 7, 7]))
x510 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x525, x510)
end = time.time()
print(end-start)
