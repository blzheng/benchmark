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
        self.conv2d32 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x93):
        x94=x93.mean((2, 3),keepdim=True)
        x95=self.conv2d32(x94)
        return x95

m = M().eval()
x93 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x93)
end = time.time()
print(end-start)
