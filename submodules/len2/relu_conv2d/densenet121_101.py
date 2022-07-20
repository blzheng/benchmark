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
        self.relu102 = ReLU(inplace=True)
        self.conv2d102 = Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x364):
        x365=self.relu102(x364)
        x366=self.conv2d102(x365)
        return x366

m = M().eval()
x364 = torch.randn(torch.Size([1, 736, 7, 7]))
start = time.time()
output = m(x364)
end = time.time()
print(end-start)
