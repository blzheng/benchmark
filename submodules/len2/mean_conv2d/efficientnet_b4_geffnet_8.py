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
        self.conv2d41 = Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x122):
        x123=x122.mean((2, 3),keepdim=True)
        x124=self.conv2d41(x123)
        return x124

m = M().eval()
x122 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x122)
end = time.time()
print(end-start)
