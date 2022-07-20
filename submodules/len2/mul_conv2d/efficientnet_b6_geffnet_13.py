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
        self.conv2d67 = Conv2d(432, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x195, x200):
        x201=operator.mul(x195, x200)
        x202=self.conv2d67(x201)
        return x202

m = M().eval()
x195 = torch.randn(torch.Size([1, 432, 28, 28]))
x200 = torch.randn(torch.Size([1, 432, 1, 1]))
start = time.time()
output = m(x195, x200)
end = time.time()
print(end-start)
