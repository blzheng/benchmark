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
        self.conv2d27 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x82, x77):
        x83=operator.mul(x82, x77)
        x84=self.conv2d27(x83)
        return x84

m = M().eval()
x82 = torch.randn(torch.Size([1, 120, 1, 1]))
x77 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x82, x77)
end = time.time()
print(end-start)
