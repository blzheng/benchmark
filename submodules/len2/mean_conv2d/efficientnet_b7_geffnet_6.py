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
        self.conv2d29 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x89):
        x90=x89.mean((2, 3),keepdim=True)
        x91=self.conv2d29(x90)
        return x91

m = M().eval()
x89 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x89)
end = time.time()
print(end-start)
