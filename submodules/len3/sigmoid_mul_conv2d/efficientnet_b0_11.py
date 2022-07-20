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
        self.sigmoid11 = Sigmoid()
        self.conv2d59 = Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x176, x172):
        x177=self.sigmoid11(x176)
        x178=operator.mul(x177, x172)
        x179=self.conv2d59(x178)
        return x179

m = M().eval()
x176 = torch.randn(torch.Size([1, 672, 1, 1]))
x172 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x176, x172)
end = time.time()
print(end-start)
