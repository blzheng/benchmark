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
        self.conv2d223 = Conv2d(1824, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x713, x708):
        x714=operator.mul(x713, x708)
        x715=self.conv2d223(x714)
        return x715

m = M().eval()
x713 = torch.randn(torch.Size([1, 1824, 1, 1]))
x708 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x713, x708)
end = time.time()
print(end-start)
