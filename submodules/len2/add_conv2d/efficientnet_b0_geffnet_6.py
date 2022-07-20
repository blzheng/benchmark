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
        self.conv2d65 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x189, x175):
        x190=operator.add(x189, x175)
        x191=self.conv2d65(x190)
        return x191

m = M().eval()
x189 = torch.randn(torch.Size([1, 192, 7, 7]))
x175 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x189, x175)
end = time.time()
print(end-start)
