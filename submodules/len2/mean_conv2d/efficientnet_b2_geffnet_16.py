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
        self.conv2d81 = Conv2d(720, 30, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x240):
        x241=x240.mean((2, 3),keepdim=True)
        x242=self.conv2d81(x241)
        return x242

m = M().eval()
x240 = torch.randn(torch.Size([1, 720, 7, 7]))
start = time.time()
output = m(x240)
end = time.time()
print(end-start)
