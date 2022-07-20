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
        self.relu91 = ReLU(inplace=True)
        self.conv2d91 = Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x324):
        x325=self.relu91(x324)
        x326=self.conv2d91(x325)
        return x326

m = M().eval()
x324 = torch.randn(torch.Size([1, 1088, 14, 14]))
start = time.time()
output = m(x324)
end = time.time()
print(end-start)
