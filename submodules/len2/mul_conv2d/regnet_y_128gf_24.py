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
        self.conv2d128 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x404, x399):
        x405=operator.mul(x404, x399)
        x406=self.conv2d128(x405)
        return x406

m = M().eval()
x404 = torch.randn(torch.Size([1, 2904, 1, 1]))
x399 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x404, x399)
end = time.time()
print(end-start)
