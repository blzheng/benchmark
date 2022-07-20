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
        self.maxpool2d2 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x23):
        x24=self.maxpool2d2(x23)
        return x24

m = M().eval()
x23 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)
