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
        self.maxpool2d5 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)

    def forward(self, x65):
        x81=self.maxpool2d5(x65)
        return x81

m = M().eval()
x65 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x65)
end = time.time()
print(end-start)
