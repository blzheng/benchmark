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

    def forward(self, x17):
        x18=self.maxpool2d1(x17)
        return x18

m = M().eval()
x17 = torch.randn(torch.Size([1, 128, 55, 55]))
start = time.time()
output = m(x17)
end = time.time()
print(end-start)
