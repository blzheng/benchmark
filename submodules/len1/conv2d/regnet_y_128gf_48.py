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
        self.conv2d48 = Conv2d(1056, 2904, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x151):
        x152=self.conv2d48(x151)
        return x152

m = M().eval()
x151 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x151)
end = time.time()
print(end-start)
