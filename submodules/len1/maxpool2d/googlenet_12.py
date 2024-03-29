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
        self.maxpool2d12 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)

    def forward(self, x186):
        x202=self.maxpool2d12(x186)
        return x202

m = M().eval()
x186 = torch.randn(torch.Size([1, 832, 7, 7]))
start = time.time()
output = m(x186)
end = time.time()
print(end-start)
