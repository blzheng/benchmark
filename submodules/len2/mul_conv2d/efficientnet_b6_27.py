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
        self.conv2d137 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x427, x422):
        x428=operator.mul(x427, x422)
        x429=self.conv2d137(x428)
        return x429

m = M().eval()
x427 = torch.randn(torch.Size([1, 1200, 1, 1]))
x422 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x427, x422)
end = time.time()
print(end-start)
