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
        self.relu5 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x23, x18):
        x24=operator.add(x23, x18)
        x25=self.relu5(x24)
        x26=self.conv2d7(x25)
        return x26

m = M().eval()
x23 = torch.randn(torch.Size([1, 64, 56, 56]))
x18 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x23, x18)
end = time.time()
print(end-start)
