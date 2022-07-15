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
        self.conv2d310 = Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x996):
        x997=self.conv2d310(x996)
        return x997

m = M().eval()
x996 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x996)
end = time.time()
print(end-start)
