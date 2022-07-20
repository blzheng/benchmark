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
        self.relu625 = ReLU6(inplace=True)
        self.conv2d38 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x108):
        x109=self.relu625(x108)
        x110=self.conv2d38(x109)
        return x110

m = M().eval()
x108 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x108)
end = time.time()
print(end-start)
