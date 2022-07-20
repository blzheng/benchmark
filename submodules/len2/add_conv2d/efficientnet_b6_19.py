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
        self.conv2d123 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x383, x368):
        x384=operator.add(x383, x368)
        x385=self.conv2d123(x384)
        return x385

m = M().eval()
x383 = torch.randn(torch.Size([1, 200, 14, 14]))
x368 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x383, x368)
end = time.time()
print(end-start)
