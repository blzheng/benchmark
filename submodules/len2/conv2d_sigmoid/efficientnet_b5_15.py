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
        self.conv2d76 = Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()

    def forward(self, x235):
        x236=self.conv2d76(x235)
        x237=self.sigmoid15(x236)
        return x237

m = M().eval()
x235 = torch.randn(torch.Size([1, 32, 1, 1]))
start = time.time()
output = m(x235)
end = time.time()
print(end-start)
