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
        self.conv2d126 = Conv2d(50, 1200, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()
        self.conv2d127 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x393, x390):
        x394=self.conv2d126(x393)
        x395=self.sigmoid25(x394)
        x396=operator.mul(x395, x390)
        x397=self.conv2d127(x396)
        return x397

m = M().eval()
x393 = torch.randn(torch.Size([1, 50, 1, 1]))
x390 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x393, x390)
end = time.time()
print(end-start)
