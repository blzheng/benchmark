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
        self.conv2d256 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid44 = Sigmoid()

    def forward(self, x824, x821):
        x825=self.conv2d256(x824)
        x826=self.sigmoid44(x825)
        x827=operator.mul(x826, x821)
        return x827

m = M().eval()
x824 = torch.randn(torch.Size([1, 96, 1, 1]))
x821 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x824, x821)
end = time.time()
print(end-start)
