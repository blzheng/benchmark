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
        self.batchnorm2d14 = BatchNorm2d(40, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d15 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x41):
        x42=self.batchnorm2d14(x41)
        x43=self.conv2d15(x42)
        return x43

m = M().eval()
x41 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x41)
end = time.time()
print(end-start)
