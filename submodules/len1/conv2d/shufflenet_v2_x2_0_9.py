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
        self.conv2d9 = Conv2d(122, 122, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x53):
        x54=self.conv2d9(x53)
        return x54

m = M().eval()
x53 = torch.randn(torch.Size([1, 122, 28, 28]))
start = time.time()
output = m(x53)
end = time.time()
print(end-start)
