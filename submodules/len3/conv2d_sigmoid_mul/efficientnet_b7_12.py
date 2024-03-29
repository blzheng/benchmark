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
        self.conv2d60 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()

    def forward(self, x186, x183):
        x187=self.conv2d60(x186)
        x188=self.sigmoid12(x187)
        x189=operator.mul(x188, x183)
        return x189

m = M().eval()
x186 = torch.randn(torch.Size([1, 20, 1, 1]))
x183 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x186, x183)
end = time.time()
print(end-start)
