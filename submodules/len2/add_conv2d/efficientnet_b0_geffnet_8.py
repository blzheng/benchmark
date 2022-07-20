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
        self.conv2d75 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x219, x205):
        x220=operator.add(x219, x205)
        x221=self.conv2d75(x220)
        return x221

m = M().eval()
x219 = torch.randn(torch.Size([1, 192, 7, 7]))
x205 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x219, x205)
end = time.time()
print(end-start)
