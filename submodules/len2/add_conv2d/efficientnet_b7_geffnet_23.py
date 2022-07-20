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
        self.conv2d137 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x410, x396):
        x411=operator.add(x410, x396)
        x412=self.conv2d137(x411)
        return x412

m = M().eval()
x410 = torch.randn(torch.Size([1, 160, 14, 14]))
x396 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x410, x396)
end = time.time()
print(end-start)
