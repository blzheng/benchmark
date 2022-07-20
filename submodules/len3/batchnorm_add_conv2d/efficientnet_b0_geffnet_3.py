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
        self.batchnorm2d14 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d25 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x71, x58):
        x72=self.batchnorm2d14(x71)
        x73=operator.add(x72, x58)
        x74=self.conv2d25(x73)
        return x74

m = M().eval()
x71 = torch.randn(torch.Size([1, 40, 28, 28]))
x58 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x71, x58)
end = time.time()
print(end-start)
