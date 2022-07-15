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
        self.conv2d143 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x455):
        x456=self.conv2d143(x455)
        return x456

m = M().eval()
x455 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x455)
end = time.time()
print(end-start)
