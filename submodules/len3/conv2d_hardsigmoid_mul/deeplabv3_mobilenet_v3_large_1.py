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
        self.conv2d17 = Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid1 = Hardsigmoid()

    def forward(self, x51, x48):
        x52=self.conv2d17(x51)
        x53=self.hardsigmoid1(x52)
        x54=operator.mul(x53, x48)
        return x54

m = M().eval()
x51 = torch.randn(torch.Size([1, 32, 1, 1]))
x48 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x51, x48)
end = time.time()
print(end-start)
