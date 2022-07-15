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
        self.conv2d98 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x291):
        x292=self.conv2d98(x291)
        return x292

m = M().eval()
x291 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x291)
end = time.time()
print(end-start)
