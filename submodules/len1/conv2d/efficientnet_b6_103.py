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
        self.conv2d103 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x322):
        x323=self.conv2d103(x322)
        return x323

m = M().eval()
x322 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x322)
end = time.time()
print(end-start)
