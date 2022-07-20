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
        self.relu69 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(432, 1008, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x238):
        x239=self.relu69(x238)
        x240=self.conv2d73(x239)
        return x240

m = M().eval()
x238 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x238)
end = time.time()
print(end-start)
