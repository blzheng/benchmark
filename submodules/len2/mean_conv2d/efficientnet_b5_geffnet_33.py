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
        self.conv2d165 = Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x492):
        x493=x492.mean((2, 3),keepdim=True)
        x494=self.conv2d165(x493)
        return x494

m = M().eval()
x492 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x492)
end = time.time()
print(end-start)
