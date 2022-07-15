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
        self.conv2d56 = Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)

    def forward(self, x169):
        x170=self.conv2d56(x169)
        return x170

m = M().eval()
x169 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
