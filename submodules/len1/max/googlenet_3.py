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
        self.maxpool2d3 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)

    def forward(self, x44):
        x60=self.maxpool2d3(x44)
        return x60

m = M().eval()
x44 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x44)
end = time.time()
print(end-start)
