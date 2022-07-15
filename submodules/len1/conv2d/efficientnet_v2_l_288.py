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
        self.conv2d288 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x927):
        x928=self.conv2d288(x927)
        return x928

m = M().eval()
x927 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x927)
end = time.time()
print(end-start)
