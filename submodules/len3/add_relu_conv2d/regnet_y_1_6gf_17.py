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
        self.relu72 = ReLU(inplace=True)
        self.conv2d94 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x281, x295):
        x296=operator.add(x281, x295)
        x297=self.relu72(x296)
        x298=self.conv2d94(x297)
        return x298

m = M().eval()
x281 = torch.randn(torch.Size([1, 336, 14, 14]))
x295 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x281, x295)
end = time.time()
print(end-start)
