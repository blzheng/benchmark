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
        self.conv2d3 = Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)

    def forward(self, x28):
        x30=self.conv2d3(x28)
        return x30

m = M().eval()
x28 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x28)
end = time.time()
print(end-start)
