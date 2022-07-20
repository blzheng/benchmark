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
        self.conv2d141 = Conv2d(960, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x417, x422):
        x423=operator.mul(x417, x422)
        x424=self.conv2d141(x423)
        return x424

m = M().eval()
x417 = torch.randn(torch.Size([1, 960, 14, 14]))
x422 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x417, x422)
end = time.time()
print(end-start)
