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
        self.relu126 = ReLU(inplace=True)
        self.conv2d126 = Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x448):
        x449=self.relu126(x448)
        x450=self.conv2d126(x449)
        return x450

m = M().eval()
x448 = torch.randn(torch.Size([1, 992, 7, 7]))
start = time.time()
output = m(x448)
end = time.time()
print(end-start)
