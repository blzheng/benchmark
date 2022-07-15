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
        self.conv2d262 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x825):
        x826=self.conv2d262(x825)
        return x826

m = M().eval()
x825 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x825)
end = time.time()
print(end-start)
