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
        self.relu0 = ReLU(inplace=True)
        self.conv2d2 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x7):
        x8=self.relu0(x7)
        x9=self.conv2d2(x8)
        return x9

m = M().eval()
x7 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x7)
end = time.time()
print(end-start)
