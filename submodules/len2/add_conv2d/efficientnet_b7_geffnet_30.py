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
        self.conv2d177 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x529, x515):
        x530=operator.add(x529, x515)
        x531=self.conv2d177(x530)
        return x531

m = M().eval()
x529 = torch.randn(torch.Size([1, 224, 14, 14]))
x515 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x529, x515)
end = time.time()
print(end-start)
