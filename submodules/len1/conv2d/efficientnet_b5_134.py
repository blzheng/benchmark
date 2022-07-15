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
        self.conv2d134 = Conv2d(1056, 1056, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=1056, bias=False)

    def forward(self, x419):
        x420=self.conv2d134(x419)
        return x420

m = M().eval()
x419 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x419)
end = time.time()
print(end-start)
