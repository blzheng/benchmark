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
        self.conv2d140 = Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x417):
        x418=x417.mean((2, 3),keepdim=True)
        x419=self.conv2d140(x418)
        return x419

m = M().eval()
x417 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x417)
end = time.time()
print(end-start)
