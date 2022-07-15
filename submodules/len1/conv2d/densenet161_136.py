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
        self.conv2d136 = Conv2d(1632, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x484):
        x485=self.conv2d136(x484)
        return x485

m = M().eval()
x484 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x484)
end = time.time()
print(end-start)
