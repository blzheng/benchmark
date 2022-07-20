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
        self.conv2d24 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x83, x77):
        x84=operator.add(x83, x77)
        x85=self.conv2d24(x84)
        return x85

m = M().eval()
x83 = torch.randn(torch.Size([1, 80, 28, 28]))
x77 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x83, x77)
end = time.time()
print(end-start)
