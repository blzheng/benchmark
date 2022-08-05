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

    def forward(self, x49, x46):
        x50=self.conv2d17(x49)
        x51=self.hardsigmoid1(x50)
        x52=operator.mul(x51, x46)
        return x52

m = M().eval()
x49 = torch.randn(torch.Size([1, 32, 1, 1]))
x46 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x49, x46)
end = time.time()
print(end-start)
