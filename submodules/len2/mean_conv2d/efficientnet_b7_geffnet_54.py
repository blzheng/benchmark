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
        self.conv2d269 = Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x804):
        x805=x804.mean((2, 3),keepdim=True)
        x806=self.conv2d269(x805)
        return x806

m = M().eval()
x804 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x804)
end = time.time()
print(end-start)
