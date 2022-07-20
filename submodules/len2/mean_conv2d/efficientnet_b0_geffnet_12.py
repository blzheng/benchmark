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
        self.conv2d62 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x181):
        x182=x181.mean((2, 3),keepdim=True)
        x183=self.conv2d62(x182)
        return x183

m = M().eval()
x181 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x181)
end = time.time()
print(end-start)
