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
        self.conv2d123 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x391):
        x392=self.conv2d123(x391)
        return x392

m = M().eval()
x391 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x391)
end = time.time()
print(end-start)
