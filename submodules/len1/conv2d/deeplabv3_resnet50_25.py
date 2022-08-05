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
        self.conv2d25 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x83):
        x84=self.conv2d25(x83)
        return x84

m = M().eval()
x83 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x83)
end = time.time()
print(end-start)
