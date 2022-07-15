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
        self.maxpool2d1 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x29):
        x30=self.maxpool2d1(x29)
        return x30

m = M().eval()
x29 = torch.randn(torch.Size([1, 192, 52, 52]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
