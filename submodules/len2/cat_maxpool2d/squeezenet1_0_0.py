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
        self.maxpool2d1 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

    def forward(self, x21, x23):
        x24=torch.cat([x21, x23], 1)
        x25=self.maxpool2d1(x24)
        return x25

m = M().eval()
x21 = torch.randn(torch.Size([1, 128, 54, 54]))
x23 = torch.randn(torch.Size([1, 128, 54, 54]))
start = time.time()
output = m(x21, x23)
end = time.time()
print(end-start)
