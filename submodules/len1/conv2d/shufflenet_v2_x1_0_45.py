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
        self.conv2d45 = Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x282):
        x283=self.conv2d45(x282)
        return x283

m = M().eval()
x282 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x282)
end = time.time()
print(end-start)
