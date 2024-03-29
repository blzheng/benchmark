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
        self.relu133 = ReLU(inplace=True)
        self.conv2d133 = Conv2d(1760, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x471):
        x472=self.relu133(x471)
        x473=self.conv2d133(x472)
        return x473

m = M().eval()
x471 = torch.randn(torch.Size([1, 1760, 14, 14]))
start = time.time()
output = m(x471)
end = time.time()
print(end-start)
