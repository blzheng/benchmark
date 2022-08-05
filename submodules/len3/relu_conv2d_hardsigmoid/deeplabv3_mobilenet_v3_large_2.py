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
        self.relu13 = ReLU()
        self.conv2d22 = Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid2 = Hardsigmoid()

    def forward(self, x65):
        x66=self.relu13(x65)
        x67=self.conv2d22(x66)
        x68=self.hardsigmoid2(x67)
        return x68

m = M().eval()
x65 = torch.randn(torch.Size([1, 32, 1, 1]))
start = time.time()
output = m(x65)
end = time.time()
print(end-start)
