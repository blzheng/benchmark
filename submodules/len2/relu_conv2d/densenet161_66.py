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
        self.relu67 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x240):
        x241=self.relu67(x240)
        x242=self.conv2d67(x241)
        return x242

m = M().eval()
x240 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x240)
end = time.time()
print(end-start)
