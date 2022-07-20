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
        self.conv2d89 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x284, x269):
        x285=operator.add(x284, x269)
        x286=self.conv2d89(x285)
        return x286

m = M().eval()
x284 = torch.randn(torch.Size([1, 160, 14, 14]))
x269 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x284, x269)
end = time.time()
print(end-start)
