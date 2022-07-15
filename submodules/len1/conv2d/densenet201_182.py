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
        self.conv2d182 = Conv2d(1632, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x645):
        x646=self.conv2d182(x645)
        return x646

m = M().eval()
x645 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x645)
end = time.time()
print(end-start)
