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
        self.relu97 = ReLU(inplace=True)
        self.conv2d103 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x337):
        x338=self.relu97(x337)
        x339=self.conv2d103(x338)
        return x339

m = M().eval()
x337 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x337)
end = time.time()
print(end-start)
