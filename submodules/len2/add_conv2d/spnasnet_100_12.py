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
        self.conv2d57 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x184, x175):
        x185=operator.add(x184, x175)
        x186=self.conv2d57(x185)
        return x186

m = M().eval()
x184 = torch.randn(torch.Size([1, 192, 7, 7]))
x175 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x184, x175)
end = time.time()
print(end-start)
