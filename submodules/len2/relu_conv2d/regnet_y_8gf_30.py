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
        self.relu40 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x168):
        x169=self.relu40(x168)
        x170=self.conv2d54(x169)
        return x170

m = M().eval()
x168 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x168)
end = time.time()
print(end-start)
