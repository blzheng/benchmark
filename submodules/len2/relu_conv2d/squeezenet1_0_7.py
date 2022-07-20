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
        self.relu22 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x55):
        x56=self.relu22(x55)
        x57=self.conv2d23(x56)
        return x57

m = M().eval()
x55 = torch.randn(torch.Size([1, 64, 13, 13]))
start = time.time()
output = m(x55)
end = time.time()
print(end-start)
