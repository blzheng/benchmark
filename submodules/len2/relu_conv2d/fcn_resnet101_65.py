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
        self.relu64 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x231):
        x232=self.relu64(x231)
        x233=self.conv2d70(x232)
        return x233

m = M().eval()
x231 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x231)
end = time.time()
print(end-start)
