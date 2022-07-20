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
        self.conv2d140 = Conv2d(1200, 50, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x418):
        x419=x418.mean((2, 3),keepdim=True)
        x420=self.conv2d140(x419)
        return x420

m = M().eval()
x418 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x418)
end = time.time()
print(end-start)
