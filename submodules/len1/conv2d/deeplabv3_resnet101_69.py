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
        self.conv2d69 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x228):
        x229=self.conv2d69(x228)
        return x229

m = M().eval()
x228 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x228)
end = time.time()
print(end-start)
