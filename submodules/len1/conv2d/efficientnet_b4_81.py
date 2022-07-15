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
        self.conv2d81 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x252):
        x253=self.conv2d81(x252)
        return x253

m = M().eval()
x252 = torch.randn(torch.Size([1, 672, 1, 1]))
start = time.time()
output = m(x252)
end = time.time()
print(end-start)
