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
        self.conv2d47 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), dilation=(2, 2), groups=672, bias=False)

    def forward(self, x137):
        x138=self.conv2d47(x137)
        return x138

m = M().eval()
x137 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x137)
end = time.time()
print(end-start)
