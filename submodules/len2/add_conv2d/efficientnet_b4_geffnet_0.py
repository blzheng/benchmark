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
        self.conv2d9 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x27, x15):
        x28=operator.add(x27, x15)
        x29=self.conv2d9(x28)
        return x29

m = M().eval()
x27 = torch.randn(torch.Size([1, 24, 112, 112]))
x15 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x27, x15)
end = time.time()
print(end-start)
