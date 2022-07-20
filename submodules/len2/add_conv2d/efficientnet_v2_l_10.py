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
        self.conv2d23 = Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x81, x75):
        x82=operator.add(x81, x75)
        x83=self.conv2d23(x82)
        return x83

m = M().eval()
x81 = torch.randn(torch.Size([1, 96, 28, 28]))
x75 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x81, x75)
end = time.time()
print(end-start)
