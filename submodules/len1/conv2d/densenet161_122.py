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
        self.conv2d122 = Conv2d(1296, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x435):
        x436=self.conv2d122(x435)
        return x436

m = M().eval()
x435 = torch.randn(torch.Size([1, 1296, 7, 7]))
start = time.time()
output = m(x435)
end = time.time()
print(end-start)
