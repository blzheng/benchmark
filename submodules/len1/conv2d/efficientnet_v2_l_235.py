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
        self.conv2d235 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x758):
        x759=self.conv2d235(x758)
        return x759

m = M().eval()
x758 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x758)
end = time.time()
print(end-start)
