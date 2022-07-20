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
        self.conv2d164 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x491):
        x492=x491.mean((2, 3),keepdim=True)
        x493=self.conv2d164(x492)
        return x493

m = M().eval()
x491 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x491)
end = time.time()
print(end-start)
