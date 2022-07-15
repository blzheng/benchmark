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
        self.conv2d65 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)

    def forward(self, x198):
        x199=self.conv2d65(x198)
        return x199

m = M().eval()
x198 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x198)
end = time.time()
print(end-start)
