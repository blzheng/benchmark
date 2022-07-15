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
        self.maxpool2d2 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x12):
        x13=self.maxpool2d2(x12)
        return x13

m = M().eval()
x12 = torch.randn(torch.Size([1, 256, 13, 13]))
start = time.time()
output = m(x12)
end = time.time()
print(end-start)
